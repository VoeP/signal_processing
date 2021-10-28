# signal_processing
Signal processing course, project done in collaboration with Rares Cristea and Bernhardsgruetter Nicole

The aim of this project was to create a program that could determine whether or not there is subretinal fluid present in an image of an eye. We did this by increasing contrast, 
finding the photoreceptor layer and removing everything in the image except the area of interest, and then doing blob detection on this area, determine the center of the blob
and calculating the distance of the center from the photoreceptor layer. In this way, it should be possible to determine if the blob is subretinal fluid, since in such a case 
the radius of the blob should be about the same as the distance of the center from the photoreceptor layer.
