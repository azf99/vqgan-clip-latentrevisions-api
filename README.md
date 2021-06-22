# mead-backend

### Performance
    * LatentRevision: Loading models takes 10-15 secs and, processing takes 2-3 secs per result
    * StyleCLIP: The model loading takes 5-8 secs, and each iteration takes on average 0.5-0.7 seconds. Satisfactory output produced after atleast 15-20 iterations(below 2.6 loss)

### Delivering Results:
    * LatentRevision: TBD
    * StyleCLIP: Processing takes, on avg 15-20 secs, maybe we can directly send the output with the request response as base64 encoded image. User can input number of iterations. Can also run on previous image.

### TODOS 
    * Add conda environment YAML file to install everything
    * Refactor LatentRevisions