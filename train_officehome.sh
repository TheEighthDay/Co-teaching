for x in {"Product","Real_World","Art","Clipart"}
do
    for y in {"Product","Real_World","Art","Clipart"}
    do
	    if [ $x !=  $y ] ;then
	            #echo $x+$y ${x:0:1}
		    #if [ $x != "Product" -o $y != "Real_World" ] && [ $x != "Clipart" -o $y != "Art" ]; then
			    #echo $x+$y ${x:0:1}
			    python train_officehome.py --model_name="KD_SRDC" --base_model_name="RN" --da_model_name="SRDC"  --base_model_checkpoint="/datastore/users/kaibin.tian/DE_experiment/base_new/RN/officehome/[${x:0:1}].pth" --da_model_checkpoint="/datastore/users/kaibin.tian/DE_experiment/da_new/log/SRDC/officehome/[${x:0:1}2${y:0:1}].pth" --source_dir=$x --target_dir=$y  --savedir="mutual_epochbeta"
                    #fi
            fi

    done
done
