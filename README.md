This repository contains the source code for the MEGC2022 with paper: Fine-grained Micro-Expression Generation based on Thin-Plate Spline and Relative AU Constraint 

And this work encouraged by [Thin-Plate-Spline-Motion-Model](https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model).

## 0. Table of Contents
* [1. Authors & Maintainers](#1-authors---maintainers)
* [2. Quantitative Results](#2-quantitative-results)
* [2. Results in GIF](#3-results-in-gif)
* [4. Run the code](#4-run-the-code)
* [5. License](#5-license)
* [6. Citation](#6-citation)

## 1. Authors & Maintainers
Sirui et al.

## 2. Quantitative Results
<img src="./sup-mat/上传1-Quantitative Results.png">

## 3. Results in GIF
Given target template face:

<img src="./sup-mat/gif/T_1-Template_Male_Asian.jpg">

the results of corresponding generation are listed below as gif.

| No.             | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    |
| :-------------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | 
|  Source samples | <img src="./sup-mat/gif/1_1-Negative_EP19_06f.gif"> | <img src="./sup-mat/gif/1_2-Positive_EP01_01f.gif"> | <img src="./sup-mat/gif/1_3-Surprise_EP01_13.gif"> | <img src="./sup-mat/gif/1_4-Negative_018_3_1.gif"> | <img src="./sup-mat/gif/1_5-Positive_022_3_3.gif"> | <img src="./sup-mat/gif/1_6-Surprise_007_7_1.gif"> | <img src="./sup-mat/gif/1_7-Negative_s11_ne_02.gif"> | <img src="./sup-mat/gif/1_8-Positive_s3_po_05.gif"> | <img src="./sup-mat/gif/1_9-Surprise_s20_sur_01.gif"> |
|  Our results    | <img src="./sup-mat/gif/2_1-Negative_EP19_06f-Template_Male_Asian.gif"> | <img src="./sup-mat/gif/2_2-Positive_EP01_01f-Template_Male_Asian .gif"> | <img src="./sup-mat/gif/2_3-Surprise_EP01_13-Template_Male_Asian.gif"> | <img src="./sup-mat/gif/2_4-Negative_018_3_1-Template_Male_Asian.gif"> | <img src="./sup-mat/gif/2_5-Positive_022_3_3-Template_Male_Asian.gif"> | <img src="./sup-mat/gif/2_6-Surprise_007_7_1-Template_Male_Asian.gif"> | <img src="./sup-mat/gif/2_7-Negative_s11_ne_02-Template_Male_Asian.gif"> | <img src="./sup-mat/gif/2_8-Positive_s3_po_05-Template_Male_Asian.gif"> | <img src="./sup-mat/gif/2_9-Surprise_s20_sur_01-Template_Male_Asian.gif"> |

## 4. Run the code
`python run.py`

## 5. License
[MIT](https://github.com/Necolizer/Facial-Prior-Based-FOMM/blob/main/LICENSE)

## 6. Citation
Sirui Zhao, Shukang Yin, Huaying Tang, Jin Rijin, Yifan Xu, Tong Xu*, Enhong Chen, Fine-grained Micro-Expression Generation based on Thin-Plate Spline and Relative AU Constraint, In Proceedings of the 30th ACM International Conference on Multimedia (ACM MM'22), Lisbon, Portugal, 2022, Accepted，DOI: 10.1145/3503161.3551597.
