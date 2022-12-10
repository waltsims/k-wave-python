# Run tests inside a Docker image
image_name='kwave-python-image'
project_dir_in_container='/k-wave-python'

docker build -t $image_name .

docker run -it \
  --volume "$(pwd)":$project_dir_in_container \
  --workdir $project_dir_in_container \
  $image_name \
  pytest    # run pytest
