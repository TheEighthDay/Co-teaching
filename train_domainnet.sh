for x in {"painting","sketch","real","clipart"}
do
    for y in {"clipart","real","painting","sketch"}
    do
	    if [ $x !=  $y ] ;then
#		    if [ $x != "painting" -o $y != "clipart" ] && [ $x != "painting" -o $y != "sketch" ] && [ $x != "real" -o $y != "painting" ] && [ $x != "real" -o $y != "sketch" ]; then
#			    echo $x+$y ${x:0:1}
			    python train_domainnet.py --model_name="KD_DDC" --base_model_name="RN" --da_model_name="DDC" --base_model_checkpoint="/datastore/users/kaibin.tian/DE_experiment/base_new_wj/RN/domainnet/[${x:0:1}].pth" --da_model_checkpoint="/datastore/users/kaibin.tian/DE_experiment/da_new/log/DDC/domainnet/[${x:0:1}2${y:0:1}].pth" --source_dir=$x --target_dir=$y --root_dir="/datastore/users/kaibin.tian/DE_experiment/domainnet/" --dataset_name="domainnet" --num_class=345 --savedir="mutual_epochbeta_repeat" --batch_size=48 --epochs=31 --scheduler_size=10
#                    fi
            fi

    done
done

