### REQUIREMENTS:
(The code has been tested on Max OS and Linux platforms.)

* This code requires installation of the VLfeat library: <p>[http://www.vlfeat.org/](http://www.vlfeat.org/)</p>
* It also uses Piotr Dollar’s toolbox <p>[https://github.com/pdollar/toolbox](https://github.com/pdollar/toolbox)</p>
* Gradient Boosted Trees implementation from Carlos Becker <p> [https://sites.google.com/site/carlosbecker/resources/gradient-boosting-boosted-trees](https://sites.google.com/site/carlosbecker/resources/gradient-boosting-boosted-trees) </p>


### RUNNING:

* Start MATLAB            
* Load video sequence to the MATLAB environment as a 3D array
* Run the following commands in MATLAB:

        vid_number = '1'
        vid_type = 'rexp'
* Run the <span style="color:purple"> detector_with_refinement.m </span> script. <p> <span style="color:red">NOTE</span>: code is not optimised, so detection may take long time </p>
    
* Output will be saved to the <span style="color:purple">"./results"</span> folder and can be viewed with <span style="color:purple">"./_supp_func/vis_stack_of_loc.m"</span> function		

### PARAMETERS:

Procedure described above will start the detection with default parameters. This parameters are described directly in the MATLAB script. You can change them there.

## OTHER:

This code implements the following work:

    @inproceedings{DBLP:conf/cvpr/RozantsevLF15,
        author    = {Artem Rozantsev and
                     Vincent Lepetit and
                     Pascal Fua},
        title     = {Flying objects detection from a single moving camera},
        booktitle = {{IEEE} Conference on Computer Vision and Pattern Recognition},
        pages     = {4128--4136},
        year      = {2015}
    }

The code was tested on the Mac OS and Linux. 	
	