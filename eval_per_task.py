'''
usage:
python eval_per_task.py --test_collection domainnet_test --source_domain real --target_domain clipart
python eval_per_task.py --test_collection officehome_test --source_domain Art --target_domain Clipart
python eval_per_task.py --test_collection officehome_test --source_domain Real_World --target_domain Clipart
'''

from common import col2anno, ROOT_PATH
from common import read_full_anno, get_pred_path, read_pred, compute_accuracy
import argparse
import os


def evaluate_per_task(test_collection, source_domain, target_domain, run, rootpath=ROOT_PATH):
    if test_collection=="officehome_test":
        networks = 'ResNet50 DDC_ResNet50 SRDC_ResNet50 KDDE_DDC_ResNet50 KDDE_SRDC_ResNet50 CT_DDC_ResNet50 CT_SRDC_ResNet50'.split()
    else:
        networks = 'ResNet50 DDC_ResNet50 KDDE_DDC_ResNet50 CT_DDC_ResNet50'.split()

    anno_name = col2anno[test_collection]
    source_gts = read_full_anno(test_collection, source_domain, anno_name, rootpath)
    target_gts = read_full_anno(test_collection, target_domain, anno_name, rootpath)  
    perf_table = []
    
    for network in networks:
        if 'ResNet50' == network:
            model_name = '_'.join([network, source_domain])
        else:
            model_name = '_'.join([network, source_domain, target_domain])
        
        source_pred_path = get_pred_path(test_collection, source_domain, model_name, run, rootpath)
        target_pred_path = get_pred_path(test_collection, target_domain, model_name, run, rootpath)
        
        source_preds = read_pred(source_pred_path)
        target_preds = read_pred(target_pred_path)
        source_accuracy = compute_accuracy(source_preds, source_gts)*100
        target_accuracy = compute_accuracy(target_preds, target_gts)*100
        expand_accuracy = (source_accuracy + target_accuracy) / 2.0
        perf_table.append((model_name, source_accuracy, target_accuracy, expand_accuracy))
        
    return perf_table
                          

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate a specific UDE task')
    parser.add_argument('--test_collection', type=str, default="domainnet_test")
    parser.add_argument('--source_domain', type=str, default="clipart")
    parser.add_argument('--target_domain', type=str, default="painting")
    parser.add_argument('--run', type=str, default="1")
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH)
    
    
    args = parser.parse_args()

    test_collection = args.test_collection
    source_domain = args.source_domain
    target_domain = args.target_domain
    rootpath = args.rootpath    
    
    print ('Read data from %s' % rootpath)
    perf_table = evaluate_per_task(test_collection, source_domain, target_domain, args.run, rootpath)
    
    print ('#Performance of the %s->%s UDE task on %s' % (source_domain, target_domain, test_collection))
    print ('#model source-domain target-domain expanded-domain')
    for record in perf_table:
        print ('%s %s' % (record[0], ' '.join(['%.2f' % x for x in record[1:]])))