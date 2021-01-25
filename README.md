# deepak_gupta_ncs_ml

Build the image using the following command

```bash
$ docker build -t gender-predictor-app:latest .
```

Run the Docker container using the command shown below.

```bash
$ docker run -d -p 5000:5000 gender-predictor-app
```

The application will be accessible at http:127.0.0.1:5000

The training data file is stored in "data/excel.csv" folder

See "example_***" files for usage examples
