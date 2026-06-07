### Aboat Git

#### 1. Our project structure
---meteorite-identification \
&ensp;|--models\
&ensp;|&emsp;&emsp;|--model 1 \
&ensp;|&emsp;&emsp;|--model 2 \
&ensp;|&emsp;&emsp;|-- ...... \
&ensp;| \
&ensp;|--  data \
&ensp;|&emsp;&emsp;|--train_images \
&ensp;|&emsp;&emsp;|--test_images \
&ensp;|&emsp;&emsp;|--dataset.py \
&ensp;|&emsp;&emsp;|-- ......\
&ensp;|\
&ensp;|-- &ensp;src \
&ensp;|&emsp;&emsp;|--data_processing_modules \
&ensp;|&emsp;&emsp;|--fine-tuning_(training) \
&ensp;|&emsp;&emsp;|--validation \
&ensp;|&emsp;&emsp;|--testing \
&ensp;|&emsp;&emsp;&emsp;&emsp;|--gen_submission_file.py \
&ensp;|\
&ensp;|--some configuration files, (e.g. .json | .yml)\
&ensp;|--readme.md \
&ensp;|--Warning.md

#### 2. Repository statements
+ Do **not** commit **models' parameters file**. Those files are too big to upload and downlaod. By the way, *Git* will prevent you from doing this.
+ Do **not** commit **images** (the data) either. The reason is the same as above.
+ For **models' parameters file** & **images**, we can make an appointment and use the *shareddata* folder to Synchronize files. However, this method is **not secure** because some people who have been assigned the same machine as ours can also access these files in the *shareddata* folder. 
+ We will **only** commit and push source codes and some required configuration documents.
+ Considering the security aspect, **models' parameters file** & **images** would be best to copy these files via an offline USB drive, a private cloud storage, ModelScope, or through our QQ group.