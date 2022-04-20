for x in {"Art","Clipart","Product","Real_World"}
do
    for y in {"Art","Clipart","Product","Real_World"}
    do
	    if [ $x !=  $y ] ;then
			    python  train.py  --config="configs/KDDE_DDC_ResNet50.yaml" --source=$x --target=$y  --dataset="officehome" --datasetroot="datasets/OfficeHome" --num_class=65 --run=2
        fi
    done
done