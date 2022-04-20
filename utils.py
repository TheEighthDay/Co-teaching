import os


model_name = "DDC_ResNet50"
dataset = "domainnet"

domains = ["clipart","painting","real","sketch"]
# domains = ["Clipart","Art","Product","Real_World"]

fr_path = os.path.join("sup_da_weights",model_name,dataset)
to_path = os.path.join("VisualSearch",dataset+"_train","Checkpoints","concepts345.txt")

for source in domains:
    for target in domains:
        if source!=target:
            if not os.path.exists("{}/{}/{}_{}_{}/run_1/".format(to_path,source,model_name,source,target)):
                os.makedirs("{}/{}/{}_{}_{}/run_1/".format(to_path,source,model_name,source,target))

            real_source = os.path.realpath("{}/[{}2{}].pth".format(fr_path,source[0],target[0]))

            os.system("ln -s -f {} {}/{}/{}_{}_{}/run_1/[{}2{}].pth".format(real_source,to_path,source,model_name,source,target,source[0],target[0]))