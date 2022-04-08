
### How "https://staging.taptapstories.dk/mask/" Works

* The project has three  endpoints as API that takes an Image as a paramater and return modified one 

#### /image endpoint:  
* The image endpoint makes images accessible through the API and also displays images that are passed to it
* Example  
```
https://staging.taptapstories.dk/mask/image?img=/static/output/grey_img_5terre.jpg

```   

#### /features endpoint:  
* The features endpoint takes two arguments, the path to an input image and the scale parameter (optional)  
* The user can scale the input image by x2 or x4 by providng scale=2 or scale=4, if no scale parameter is provided, the input image is saved as it as  
* Images are saved inside static/output/ directory with a prefix that describes the operation performed on the image  
* Example  
```
https://staging.taptapstories.dk/mask/features?image=https://www.w3schools.com/css/img_5terre.jpg&scale=2
```  
 
* This returns a json with the extracted features along with a URL of the masked photo

#### /filters endpoint:  
* The filters endpoint takes two arguments, the path to an input image and the filter type 
* Example  
```
https://staging.taptapstories.dk/mask/filter?image=https://www.w3schools.com/css/img_5terre.jpg&filter=grey
```  
* This endpoint also returns a json with a URL of the output image  


#### /segment endpoint:  
* The segment endpoint takes image url then return json with the features of each face in that image and url of the cropped faces of that image each face is in sperate image

* Example  
```
https://staging.taptapstories.dk/mask/segment?image=https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8ZmFjZXxlbnwwfHwwfHw%3D&w=1000&q=80
```  

* This endpoint returns a json with a each face passed to features endpoint 