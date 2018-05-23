# Gender classification and safe search / nsfw web service

> Web exposed convolutional neural network microservice validating the appropriacy of images

This was for a side project required a machine learning microservice that identified both gender and whether the image was suitable for work. The initial models will be basic and available from https://github.com/BVLC/caffe/wiki/Model-Zoo

Modelling used:
- __Caffe__ - More specifically the blvc/caffe base image - cpu flavour
- __gender_classification__ - Generic Gender Model from Gil Levi and Tal Hassner - https://www.openu.ac.il/home/hassner/projects/cnn_agegender/
- __nsfw_classification__ - Yahoo NSFW - https://github.com/yahoo/open_nsfw

I will experiment with these suitability of these models before using other models / build my own. I will more than likely look at tensorflow as part of my learnings also.
I will also add parameters to the web services to choose which classification services are completed as part of the request.

### Installation
```bash
docker run -p 3005:80 -e S3_BUCKET=$S3_BUCKET -e AWS_ACCESS_KEY_ID=$S3_ACCESS -e AWS_SECRET_ACCESS_KEY=$S3_SECRET dangarfield:safe-search
```
You can pass aws environment variables including a s3 bucket name and the pulling of the images will go directly through AWS (if you are in AWS) rather than through an internet gateway.

### Usage

> GET /?image=[url]
/ or any path


### Gender Examples (I'm not adding NSFW ones here):

##### Daenerys Targaryen

![Daenerys Targaryen](https://pbs.twimg.com/profile_images/732527469383823360/PUHHZbj4_400x400.jpg "Daenerys Targaryen")
Request
`?image=https://pbs.twimg.com/profile_images/732527469383823360/PUHHZbj4_400x400.jpg`
Response
```json
{
    "gender": "Female",
    "image": "https://pbs.twimg.com/profile_images/732527469383823360/PUHHZbj4_400x400.jpg?1527080616.94",
    "nsfw": 0.04,
    "timings": {
        "download": 0.16,
        "gender": 1.31,
        "total": 1.66,
        "nsfw": 0.19
    }
}
```
##### Jaime Lannister
![Jaime Lannister](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Nikolaj_Coster-Waldau_by_Gage_Skidmore.jpg/170px-Nikolaj_Coster-Waldau_by_Gage_Skidmore.jpg "Jaime Lannister")

Request
`?image=https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Nikolaj_Coster-Waldau_by_Gage_Skidmore.jpg/170px-Nikolaj_Coster-Waldau_by_Gage_Skidmore.jpg`

Response
```json
{
    "gender": "Male",
    "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Nikolaj_Coster-Waldau_by_Gage_Skidmore.jpg/170px-Nikolaj_Coster-Waldau_by_Gage_Skidmore.jpg?1527079410.33",
    "nsfw": 0.01,
    "timings": {
        "download": 0.13,
        "gender": 1.42,
        "total": 1.75,
        "nsfw": 0.21
    }
}
```

