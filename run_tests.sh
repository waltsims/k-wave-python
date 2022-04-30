# Run tests inside a Docker image
image_name='kwave-python-image'
project_dir_in_container='/k-wave-python'

docker build -t $image_name .

docker run -it \
  --volume $(pwd):$project_dir_in_container \  # attach project folder into container
  --workdir $project_dir_in_container \        # set working directory to project folder
  $image_name \
  pytest    # run pytest
