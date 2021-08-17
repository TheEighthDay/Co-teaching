
for x in {"Product","Real_World","Art","Clipart"}
do
    for y in {"Product","Real_World","Art","Clipart"}
    do
	    if [ $x !=  $y ] ;then
			    python predict.py --model_name="KD_SRDC" --source_dir=$x --target_dir=$y 
			    python predict.py --model_name="TMT_SRDC" --source_dir=$x --target_dir=$y 
			    python predict.py --model_name="SRDC" --source_dir=$x --target_dir=$y 
			    python predict.py --model_name="RN" --source_dir=$x --target_dir=$y 
            fi

    done
done
