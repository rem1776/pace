import abc
import dataclasses
import warnings
import os
from datetime import datetime, timedelta
from typing import List, Optional, Union
from mpi4py import MPI

import numpy as np

from ndsl import Quantity
from ndsl.constants import RGRAV, Z_DIM, Z_INTERFACE_DIM
from ndsl.dsl.dace.orchestration import dace_inhibitor
from ndsl.dsl.typing import Float
from ndsl.filesystem import get_fs
from ndsl.grid import GridData
from ndsl.monitor import Monitor, ZarrMonitor
from ndsl.monitor.netcdf_monitor import NetCDFMonitor
from ndsl.typing import Communicator
from pace.state import DriverState
from pyfv3 import DycoreState

from pyfms import diag_manager, fms, mpp_domains

# ???
from ndsl.constants import (
    X_DIM,
    X_INTERFACE_DIM,
    Y_DIM,
    Y_INTERFACE_DIM,
    Z_DIM,
    Z_INTERFACE_DIM,
)


try:
    import zarr.storage as zarr_storage
except ModuleNotFoundError:
    zarr_storage = None


class Diagnostics(abc.ABC):
    @abc.abstractmethod
    def store(self, time: Union[datetime, timedelta], state: DriverState): ...

    @abc.abstractmethod
    def store_grid(self, grid_data: GridData): ...

    @abc.abstractmethod
    def cleanup(self): ...


@dataclasses.dataclass
class ZSelect:
    level: int
    names: List[str]

    def select_data(self, state: DycoreState):
        output = {}
        for name in self.names:
            if name not in state.__dict__.keys():
                raise ValueError(f"Invalid state variable {name} for level select")
            assert len(getattr(state, name).dims) > 2
            if getattr(state, name).dims[2] != (Z_DIM or Z_INTERFACE_DIM):
                raise ValueError(
                    f"z_select only works for state variables with dimension (x, y, z). \
                        \n {name} has dimension {getattr(state, name).dims}"
                )
            var_name = f"{name}_z{self.level}"
            output[var_name] = Quantity(
                getattr(state, name).data[:, :, self.level],
                dims=getattr(state, name).dims[0:2],
                origin=getattr(state, name).origin[0:2],
                extent=getattr(state, name).extent[0:2],
                units=getattr(state, name).units,
            )
        return output


@dataclasses.dataclass(frozen=True)
class DiagnosticsConfig:
    """
    Attributes:
        path: directory to save diagnostics if given, otherwise no diagnostics
            will be stored
        output_format: one of "zarr" or "netcdf", be careful when using the "netcdf"
            format as this requires all diagnostics to be stored in memory on the
            root rank before saving, which can cause out-of-memory errors if the
            global data size or number of variables is too large
        time_chunk_size: number of timesteps stored in each netcdf file, only used if
            output_format is "netcdf"
        names: state variables to save as diagnostics
        derived_names: derived diagnostics to save
        z_select: save a vertical slice of a 3D state
    """

    path: Optional[str] = None
    output_format: str = "zarr"
    time_chunk_size: int = 1
    names: List[str] = dataclasses.field(default_factory=list)
    derived_names: List[str] = dataclasses.field(default_factory=list)
    z_select: List[ZSelect] = dataclasses.field(default_factory=list)
    precision: str = "Float"

    def __post_init__(self):
        if (len(self.names) > 0 or len(self.derived_names) > 0) and self.path is None:
            raise ValueError(
                "DiagnosticsConfig.path must be given to enable diagnostics"
            )
        if self.output_format not in ["zarr", "netcdf", "diag_manager"]:
            raise ValueError(
                "output_format must be one of 'zarr' or 'netcdf', "
                f"got {self.output_format}"
            )
        if self.precision not in ["Float", "float32", "float64"]:
            raise ValueError(
                "precision must be one of 'Float', 'float32', or 'float64"
                f"got {self.precision}"
            )

    def diagnostics_factory(self, communicator: Communicator) -> Diagnostics:
        """
        Create a diagnostics object.

        Args:
            communicator: provides global communication e.g. to gather state
                or to coordinate filesystem access between ranks
        """
        if self.path is None:
            diagnostics: Diagnostics = NullDiagnostics()
        else:
            fs = get_fs(self.path)
            if not fs.exists(self.path):
                fs.makedirs(self.path, exist_ok=True)
            if self.output_format == "zarr":
                store = zarr_storage.DirectoryStore(path=self.path)
                monitor: Monitor = ZarrMonitor(
                    store=store,
                    partitioner=communicator.partitioner,
                    mpi_comm=communicator.comm,
                )
            elif self.output_format == "netcdf":
                if self.precision == "Float":
                    precision = Float
                elif self.precision == "float32":
                    precision = np.float32
                elif self.precision == "float64":
                    precision = np.float64
                monitor = NetCDFMonitor(
                    path=self.path,
                    communicator=communicator,
                    time_chunk_size=self.time_chunk_size,
                    precision=precision,
                )
            elif self.output_format == "diag_manager":
                if self.precision == "Float":
                    precision = Float
                elif self.precision == "float32":
                    precision = np.float32
                elif self.precision == "float64":
                    precision = np.float64
                diagnostics = DiagManagerDiagnostics(
                    names=self.names,
                    derived_names=self.derived_names,
                )
                return diagnostics
            else:
                raise ValueError(
                    "output_format must be one of 'zarr' or 'netcdf', "
                    f"got {self.output_format}"
                )
            diagnostics = MonitorDiagnostics(
                monitor=monitor,
                names=self.names,
                derived_names=self.derived_names,
                z_select=self.z_select,
            )
        return diagnostics


class MonitorDiagnostics(Diagnostics):
    """Diagnostics that save to a sympl-style Monitor."""

    def __init__(
        self,
        monitor: Monitor,
        names: List[str],
        derived_names: List[str],
        z_select: List[ZSelect],
    ):
        """
        Args:
            monitor: a sympl-style Monitor object
            names: list of names of diagnostics to save
            derived_names: list of names of derived diagnostics to save
        """
        self.names = names
        self.derived_names = derived_names
        self.z_select = z_select
        self.monitor = monitor

    @dace_inhibitor
    def store(self, time: Union[datetime, timedelta], state: DriverState):
        monitor_state = {"time": time}
        for name in self.names:
            try:
                quantity = getattr(state.dycore_state, name)
            except AttributeError:
                quantity = getattr(state.physics_state, name)
            monitor_state[name] = quantity
        derived_state = self._get_derived_state(state)
        level_select_state = self._get_z_select_state(state.dycore_state)
        monitor_state.update(derived_state)
        monitor_state.update(level_select_state)
        self.monitor.store(monitor_state)

    def _get_derived_state(self, state: DriverState):
        output = {}
        if len(self.derived_names) > 0:
            for name in self.derived_names:
                if name.startswith("column_integrated_"):
                    tracer = name[len("column_integrated_") :]
                    output[name] = _compute_column_integral(
                        name,
                        getattr(state.dycore_state, tracer),
                        state.dycore_state.delp,
                    )
                else:
                    warnings.warn(f"{name} is not a supported diagnostic variable.")
        return output

    def _get_z_select_state(self, state: DycoreState):
        z_select_state = {}
        for zselect in self.z_select:
            z_select_state.update(zselect.select_data(state))
        return z_select_state

    def store_grid(self, grid_data: GridData):
        zarr_grid = {
            "lat": grid_data.lat,
            "lon": grid_data.lon,
            "lon_agrid": grid_data.lon_agrid,
            "lat_agrid": grid_data.lat_agrid,
        }
        for k, v in zarr_grid.items():
            self.monitor.store_constant({k: v})

    def cleanup(self):
        self.monitor.cleanup()


class NullDiagnostics(Diagnostics):
    """Diagnostics that do nothing."""

    def store(self, time: Union[datetime, timedelta], state: DriverState):
        pass

    def store_grid(self, grid_data: GridData):
        pass

    def cleanup(self):
        pass


def _compute_column_integral(name: str, q_in: Quantity, delp: Quantity):
    """
    Compute column integrated mixing ratio (e.g., total liquid water path)

    Args:
        name: name of the tracer
        q_in: tracer mixing ratio
        delp: pressure thickness of atmospheric layer
    """
    assert len(q_in.dims) > 2
    if q_in.dims[2] != Z_DIM:
        raise NotImplementedError(
            "this function assumes the z-dimension is the third dimension"
        )
    k_slice = slice(q_in.origin[2], q_in.origin[2] + q_in.extent[2])
    column_integral = Quantity(
        RGRAV
        * q_in.np.sum(q_in.data[:, :, k_slice] * delp.data[:, :, k_slice], axis=2),
        dims=tuple(q_in.dims[:2]) + tuple(q_in.dims[3:]),
        origin=q_in.metadata.origin[0:2],
        extent=(q_in.metadata.extent[0], q_in.metadata.extent[1]),
        units="kg/m**2",
    )
    return column_integral


class DiagManagerDiagnostics(Diagnostics):
    """Diagnostics that use FMS's diag_manager from pyFMS."""

    # idk what im doing here tbh
    initialized: bool
    field_ids: dict
    precision: str

    def __init__(
        self,
        names: List[str],
        derived_names: List[str],
    ):
        """
        Args:
            names: list of names of diagnostics to save
            derived_names: list of names of derived diagnostics to save
        """
        self.names = names
        self.derived_names = derived_names
        self.field_ids = {}
        self.initialized = False
        # this env var is set by ndsl
        self.precision = "float" + os.environ["GT4PY_LITERAL_FLOAT_PRECISION"]
        fms.init(localcomm=MPI.COMM_WORLD.py2f(), calendar_type=fms.NOLEAP)

    @dace_inhibitor
    def store(self, time: Union[datetime, timedelta], state: DriverState):
        """
        Stores data from any given names via the pyFMS diag_manager
        This requires a diag_table.yaml file to be in the run directory (for now? at least)
        """

        if not self.initialized:
            self._mpp_diag_manager_init(state, time)
            self.initialized = True

        # get each variable from the dycore state and pass in it's data to the diag_manager
        # needs to be contiguous and transposed since its fortran
        for name in self.names:
           field_id = self.field_ids[name]
           field_quantity = getattr(state.dycore_state, name)
           print("**************************** calling send data*******************")
           diag_manager.send_data(diag_field_id=field_id, field=np.ascontiguousarray(field_quantity.data.transpose()))
           diag_manager.send_complete(field_id)
           diag_manager.advance_field_time(field_id)

        # TODO this stuff is more specific to pace's existing diagnostics, will need to decide how to handle it
        #derived_state = self._get_derived_state(state)
        #level_select_state = self._get_z_select_state(state.dycore_state)
        #monitor_state.update(derived_state)
        #monitor_state.update(level_select_state)
        #self.monitor.store(monitor_state)


    # called at the end to save entire grid state
    def store_grid(self, grid_data: GridData):
        pass

    def cleanup(self):
        diag_manager.end()

    # handles the initial fms/mpp/diag_manager initializations
    def _mpp_diag_manager_init(self, state: DriverState, time):

        # below returns different numbers than what is set by nx_tile
        #nx, ny = state.grid_data.lat.shape
        (x_interface, y_interface) = state.grid_data.lat.extent
        # TODO prob not always true
        x = x_interface - 2
        y = y_interface - 2

        # set up mpp domain
        global_indices = [0, x, 0, y]
        npes = MPI.COMM_WORLD.Get_size()
        layout = [1, npes]
        io_layout = [1, 1]
        domain = mpp_domains.define_domains(
            global_indices=global_indices,
            layout=layout,
        )
        mpp_domains.define_io_domain(
            domain_id=domain.domain_id,
            io_layout=io_layout,
        )
        diag_manager.init(diag_model_subset=diag_manager.DIAG_ALL)
        mpp_domains.set_current_domain(domain_id=domain.domain_id)

        # set up axes for our data
        x = np.arange(x, dtype=self.precision)
        y = np.arange(y, dtype=self.precision)
        x_interface = np.arange(x_interface, dtype=self.precision)
        y_interface = np.arange(y_interface, dtype=self.precision)
        u_quantity = getattr(state.dycore_state, "u") # TODO find a better way to get the z value
        z_size = u_quantity.shape[2] - 1
        z = np.arange(z_size, dtype=self.precision)
        id_x = diag_manager.axis_init(
            name="x",
            long_name="x",
            axis_data=x,
            cart_name="x",
            domain_id=domain.domain_id,
            set_name="atm",
            units="radians"
        )
        id_y = diag_manager.axis_init(
            name="y",
            long_name="y",
            axis_data=y,
            cart_name="y",
            domain_id=domain.domain_id,
            set_name="atm",
            units="radians"
        )
        id_z = diag_manager.axis_init(
            name="z",
            long_name="z",
            axis_data=z,
            cart_name="z",
            domain_id=domain.domain_id,
            set_name="atm",
            not_xy=True,
            units="radians"
        )
        id_x_interface = diag_manager.axis_init(
            name="x_interface",
            long_name="x_interface",
            axis_data=x_interface,
            cart_name="x_interface",
            domain_id=domain.domain_id,
            set_name="atm",
            not_xy=True,
            units="radians"
        )
        id_y_interface = diag_manager.axis_init(
            name="y_interface",
            long_name="y_interface",
            axis_data=y_interface,
            cart_name="y_interface",
            domain_id=domain.domain_id,
            set_name="atm",
            not_xy=True,
            units="radians"
        )
        axis_ids = {
            "x": id_x,
            "y": id_y,
            "z": id_z,
            "x_interface": id_x_interface,
            "y_interface": id_y_interface,
        }

        # current time is the start time
        diag_manager.set_field_init_time(
            year=time.year,
            month=time.month,
            day=time.day,
            hour=time.hour,
            minute=time.minute,
            second=time.second,
        )
        print(f"diag manager init time set as {time.year} {time.month} {time.day} {time.hour} {time.minute} {time.second}")

        # TODO get the proper end time from the driver, this is hardcoded for the baroclinic test
        diag_manager.set_time_end(
            year=time.year,
            month=time.month,
            day=time.day,
            hour=time.hour,
            minute=15,
            second=0,
        )
        print(f"diag manager end time set as {time.year} {time.month} {time.day} {time.hour} 15 0")

        for name in self.names:
            # get the quantity for each requested name from the dycore/physics states
            try:
                quantity = getattr(state.dycore_state, name)
            except AttributeError:
                quantity = getattr(state.physics_state, name)
            # get its axis id numbers and register the field
            var_axis_ids = list(map(lambda dimname: axis_ids[dimname], quantity.dims))
            # TODO should reverse ordering to pass into fortran, but this breaks the send_data call (with/without transposed data)
            #var_axis_ids.reverse()

            field_id = diag_manager.register_field_array(
                module_name="atm_mod",
                field_name=name,
                long_name=name,
                axes=var_axis_ids,
                dtype=self.precision,
                units=quantity.units
            )
            self.field_ids[name] = field_id
            # TODO set the timestep, hardcoding for now
            diag_manager.set_field_timestep(
               diag_field_id=field_id,
               dseconds=225,
               ddays=0,
               dticks=0,
            )
        self.initialized = True

