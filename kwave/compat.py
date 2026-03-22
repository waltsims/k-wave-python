"""Backward-compatibility bridge: convert legacy options classes to kspaceFirstOrder() kwargs."""


def options_to_kwargs(simulation_options=None, execution_options=None):
    """Convert legacy SimulationOptions + SimulationExecutionOptions to kspaceFirstOrder() kwargs."""
    kwargs = {}

    if simulation_options is not None:
        opts = simulation_options
        if opts.pml_auto:
            kwargs["pml_size"] = "auto"
        elif opts.pml_x_size is not None:
            sizes = [opts.pml_x_size]
            if opts.pml_y_size is not None:
                sizes.append(opts.pml_y_size)
            if opts.pml_z_size is not None:
                sizes.append(opts.pml_z_size)
            # Collapse to scalar when all set values match (safe for any ndim)
            kwargs["pml_size"] = sizes[0] if len(set(sizes)) == 1 else tuple(sizes)
        elif opts.pml_size is not None:
            s = opts.pml_size
            if hasattr(s, "__len__"):
                kwargs["pml_size"] = int(s[0]) if len(s) == 1 else tuple(int(x) for x in s)
            else:
                kwargs["pml_size"] = int(s)

        if opts.pml_x_alpha is not None:
            alphas = [opts.pml_x_alpha]
            if opts.pml_y_alpha is not None:
                alphas.append(opts.pml_y_alpha)
            if opts.pml_z_alpha is not None:
                alphas.append(opts.pml_z_alpha)
            kwargs["pml_alpha"] = alphas[0] if len(set(alphas)) == 1 else tuple(alphas)
        elif opts.pml_alpha is not None:
            kwargs["pml_alpha"] = opts.pml_alpha

        kwargs["use_sg"] = opts.use_sg
        kwargs["use_kspace"] = opts.use_kspace
        kwargs["smooth_p0"] = opts.smooth_p0
        if opts.data_path is not None:
            kwargs["data_path"] = opts.data_path
        if opts.save_to_disk_exit:
            kwargs["save_only"] = True

    if execution_options is not None:
        opts = execution_options
        if opts.backend == "python":
            kwargs["backend"] = "python"
            kwargs["device"] = "gpu" if opts.is_gpu_simulation else "cpu"
        elif opts.backend == "CUDA":
            kwargs["backend"] = "cpp"
            kwargs["device"] = "gpu"
        elif opts.backend == "OMP":
            kwargs["backend"] = "cpp"
            kwargs["device"] = "cpu"
        elif opts.is_gpu_simulation:
            kwargs["backend"] = "cpp"
            kwargs["device"] = "gpu"
        else:
            kwargs["backend"] = "cpp"
            kwargs["device"] = "cpu"

        if opts.verbose_level == 0 and not opts.show_sim_log:
            kwargs["quiet"] = True
        elif opts.verbose_level >= 2:
            kwargs["debug"] = True

        if opts.num_threads_explicit:
            kwargs["num_threads"] = opts.num_threads
        if opts.device_num is not None:
            kwargs["device_num"] = opts.device_num

    return kwargs
