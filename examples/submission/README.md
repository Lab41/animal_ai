# Submission folder
This folder contains all the necessary components for building and testing a submission.

* `agent.py` : This script implements an `Agent` class that contains all the functionality for loading an agent and having it interact with an environment.  Read the file for more details on the methods the class needs.
* `Dockerfile` : The specifications for the docker image that are submitted.  Don't alter anything outside the block that says 'YOUR COMMANDS GO HERE'.  It already includes a command for installing packages from `/data/requirements.txt` and including `data/` in the image's path so noone else should need to alter it.
* `data/` :  This directory should contain all data required by `agent.py` outside of packages installed by the Dockerfile.
  * `requirements.txt` : Include all packages to be installed by pip that aren't included in `animalai` or `animalai-train`
  * `trained_models/` : Stored trained models here, see `agent.py` for an example of how to use absolute paths to load saved models.
  * `a3c_src/` : Source code module for A3C.  If you wish to include another module like this, make sure to include a `__init_.py` file in the folder like is done here.
* `test_submission` : This directory contains utilities and data for testing a docker image.  Make sure to add the default linux executable `AnimalAI.xx86_64` and the accompanying `AnimalAI_Data/` to `test_submission/env/`.  Don't alter anything else.

For building, testing, and submitting an image, read the [submission documentation](../../documentation/submission.md)
