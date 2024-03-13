import numpy as np

from kwave.ktransducer import NotATransducer


def retract_transducer_grid_size(source, sensor, retract_size, pml_inside: bool):
    # resize the transducer object if the grid has been expanded
    is_source_kwave_transducer = isinstance(source, NotATransducer)
    is_sensor_kwave_transducer = isinstance(sensor, NotATransducer)
    retract_size = np.array(retract_size)

    if not pml_inside and (is_source_kwave_transducer or is_sensor_kwave_transducer):
        # check if the sensor is a transducer
        if is_sensor_kwave_transducer:
            # retract the transducer mask
            sensor.retract_grid(retract_size)

        # check if the source is a transducer, and if so, and different
        # transducer to the sensor
        if is_source_kwave_transducer and not (is_sensor_kwave_transducer and sensor == source):
            # retract the transducer mask
            source.retract_grid(retract_size)
