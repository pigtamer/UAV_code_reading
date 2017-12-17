### REQUIREMENTS:

* This code requires installation of the liblinear package for 'SVM' training: <p> [https://www.csie.ntu.edu.tw/~cjlin/liblinear/](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) </p>
* It also uses Piotr Dollar’s toolbox: <p> [https://github.com/pdollar/toolbox](https://github.com/pdollar/toolbox) </p>
* Gradient Boosted Trees implementation from Carlos Becker <p> [https://sites.google.com/site/carlosbecker/resources/gradient-boosting-boosted-trees](https://sites.google.com/site/carlosbecker/resources/gradient-boosting-boosted-trees) </p>

### FUNCTIONS:

<span style="color:purple">export_reg_data_using_hbt.m</span> - exports data from the video files

```coffeescript
Modify:
    [db_path]       - path to the root directory where the data is stored (line 3)
                    - the directory should contain folders:
                        1. annotations
                        2. videos
    [object]        - type of object (line 4)
                        1. drones
                        2. planes
    [data2gen]      - type of data to generate (line 5)
                        1. 'pos' - positives
                        2. 'neg' - negatives
```

<span style="color:purple">test_on_data_from_file_HBT.m</span> - aligns the bounding boxes with the data using Boosted Trees motion compensation method.

```coffeescript
Modify:
    [line 6]        - to point at the location of the Vlfeat library
    [line 7]        - to point at the location of the Gradient Boosted Trees implementation
    [pth_code]      - path to the root directory where the data is stored
```

<span style="color:purple">train_hog3d_model.m</span> - trains model based on extracted data.

```coffeescript
Input:
    [pos_data]      - array or positive samples: NxM, where N is the number of samples
                      and M = WxHxT of the spatio-temporal cube (st-cube)
    [neg_data]      - array or negative samples: NxM
    [pref]          - some label added to the saved file (can be set arbitrary)
    [method]        - choose:
                        1. 'svm' - Support Vector Machine detector
                        2. 'btr' - Boosted Trees detector
    [numIters]      - number of iterations for the Boosted trees algorithm 
                      (only used if you choose 'btr' before otherwise set it to [])
    [si]            - height of the st-cube
    [sj]            - width of the st-cube
    [st]            - temporal depth of the st-cube
Output:
    [pl_svm]        - trained model of either 'svm' or 'btr' type
    [pref]          - some label added to the saved file
```

### OTHER:

This code implements the following work:

    @article{DBLP:journals/pami/RozantsevLF17,
        author    = {Artem Rozantsev and
                     Vincent Lepetit and
                     Pascal Fua},
        title     = {Detecting Flying Objects Using a Single Moving Camera},
        journal   = {{IEEE} Transactions on Pattern Analysis Machine Intelligence},
        volume    = {39},
        number    = {5},
        pages     = {879--892},
        year      = {2017}
    }

The code was tested on the Mac OS. 



		
	