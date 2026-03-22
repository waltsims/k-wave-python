"""
Backward-compatibility bridge for migrating from legacy options classes
to the unified kspaceFirstOrder() kwargs API.

Usage:
    from kwave.compat import options_to_kwargs
    kwargs = options_to_kwargs(simulation_options, execution_options)
    result = kspaceFirstOrder(kgrid, medium, source, sensor, **kwargs)
"""


def options_to_kwargs(simulation_options=None, execution_options=None):
    """
    Convert legacy SimulationOptions + SimulationExecutionOptions to kwargs
    for kspaceFirstOrder().

    Args:
        simulation_options: Legacy SimulationOptions instance (or None)
        execution_options: Legacy SimulationExecutionOptions instance (or None)

    Returns:
        dict of kwargs suitable for kspaceFirstOrder()
    """
    kwargs = {}

    if simulation_options is not None:
        _extract_sim_options(kwargs, simulation_options)

    if execution_options is not None:
        _extract_exec_options(kwargs, execution_options)

    return kwargs


def _extract_sim_options(kwargs, opts):
    """Extract simulation options into kwargs dict."""
    # PML size
    if opts.pml_auto:
        kwargs["pml_size"] = "auto"
    elif opts.pml_x_size is not None:
        sizes = [opts.pml_x_size]
        if opts.pml_y_size is not None:
            sizes.append(opts.pml_y_size)
        if opts.pml_z_size is not None:
            sizes.append(opts.pml_z_size)
        if len(sizes) == 1:
            kwargs["pml_size"] = sizes[0]
        else:
            kwargs["pml_size"] = tuple(sizes)
    elif opts.pml_size is not None:
        if hasattr(opts.pml_size, "__len__") and len(opts.pml_size) == 1:
            kwargs["pml_size"] = int(opts.pml_size[0])
        elif hasattr(opts.pml_size, "__len__"):
            kwargs["pml_size"] = tuple(int(x) for x in opts.pml_size)
        else:
            kwargs["pml_size"] = int(opts.pml_size)

    # PML alpha
    if opts.pml_x_alpha is not None:
        alphas = [opts.pml_x_alpha]
        if opts.pml_y_alpha is not None:
            alphas.append(opts.pml_y_alpha)
        if opts.pml_z_alpha is not None:
            alphas.append(opts.pml_z_alpha)
        # Collapse to scalar if all the same
        if len(set(alphas)) == 1:
            kwargs["pml_alpha"] = alphas[0]
        else:
            kwargs["pml_alpha"] = tuple(alphas)
    elif opts.pml_alpha is not None:
        kwargs["pml_alpha"] = opts.pml_alpha

    # Boolean options
    kwargs["use_sg"] = opts.use_sg
    kwargs["use_kspace"] = opts.use_kspace
    kwargs["smooth_p0"] = opts.smooth_p0

    # Data path
    if opts.data_path is not None:
        kwargs["data_path"] = opts.data_path

    # Save only
    if opts.save_to_disk_exit:
        kwargs["save_only"] = True


def _extract_exec_options(kwargs, opts):
    """Extract execution options into kwargs dict."""
    # Backend
    if opts.backend == "native":
        kwargs["backend"] = "native"
        kwargs["use_gpu"] = opts.is_gpu_simulation or False
    elif opts.backend == "CUDA":
        kwargs["backend"] = "cpp"
        kwargs["use_gpu"] = True
    elif opts.backend == "OMP":
        kwargs["backend"] = "cpp"
        kwargs["use_gpu"] = False
    elif opts.is_gpu_simulation:
        kwargs["backend"] = "cpp"
        kwargs["use_gpu"] = True
    else:
        kwargs["backend"] = "cpp"
        kwargs["use_gpu"] = False

    # Verbosity
    if opts.verbose_level == 0 and not opts.show_sim_log:
        kwargs["quiet"] = True
    elif opts.verbose_level >= 2:
        kwargs["debug"] = True

    # Threading — only forward if user explicitly set it (not the auto-detected cpu_count default)
    import os

    cpu_count = os.cpu_count()
    if opts._num_threads is not None and opts._num_threads != cpu_count:
        kwargs["num_threads"] = opts.num_threads

    # GPU device
    if opts.device_num is not None:
        kwargs["device_num"] = opts.device_num
