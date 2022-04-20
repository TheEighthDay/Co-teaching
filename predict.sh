for x in {"Art","Clipart","Product","Real_World"}
do
    for y in {"Art","Clipart","Product","Real_World"}
    do
	    if [ $x !=  $y ] ;then
			    # python  predict.py  --config="configs/CT_SRDC_ResNet50.yaml" --source=$x --target=$y  --dataset="officehome" --datasetroot="datasets/OfficeHome" --num_class=65 --run=1 
                # python  predict.py  --config="configs/CT_DDC_ResNet50.yaml" --source=$x --target=$y  --dataset="officehome" --datasetroot="datasets/OfficeHome" --num_class=65 --run=1 
                # python  predict.py  --config="configs/KDDE_SRDC_ResNet50.yaml" --source=$x --target=$y  --dataset="officehome" --datasetroot="datasets/OfficeHome" --num_class=65 --run=1 
                python  predict.py  --config="configs/KDDE_DDC_ResNet50.yaml" --source=$x --target=$y  --dataset="officehome" --datasetroot="datasets/OfficeHome" --num_class=65 --run=1
        fi
    done
done
# for x in {"clipart","painting","real","sketch"}
# do
#     for y in {"clipart","painting","real","sketch"}
#     do
# 	    if [ $x !=  $y ] ;then
# 			    python  predict.py  --config="configs/CT_DDC_ResNet50.yaml" --source=$x --target=$y  --dataset="domainnet" --datasetroot="datasets/domainnet" --num_class=345 --run=1 --annotation="concepts345.txt"
#         fi
#     done
# done
