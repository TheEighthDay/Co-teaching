'''
usage:
python eval_all_tasks.py --test_collection domainnet_test
python eval_all_tasks.py --test_collection officehome_test
'''
import numpy as np
import argparse

from common import ROOT_PATH, read_domain_list
from eval_per_task import evaluate_per_task 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate all UDE tasks and report overall performance')
    parser.add_argument('--test_collection', type=str, default="officehome_test")
    parser.add_argument('--run', type=str, default="1")
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH)
    
    args = parser.parse_args()

    test_collection = args.test_collection
    rootpath = args.rootpath    
    
    print ('Read data from %s' % rootpath)
    
    domains = read_domain_list(test_collection, rootpath)
    ude_tasks = []
    for sd in domains:
        for td in domains:
            if sd != td:
                ude_tasks.append((sd, td))
    
    print ('Number of UDE tasks: %d' % len(ude_tasks))
    if test_collection=="officehome_test":
        methods = 'ResNet50 DDC_ResNet50 SRDC_ResNet50 KDDE_DDC_ResNet50 KDDE_SRDC_ResNet50 CT_DDC_ResNet50 CT_SRDC_ResNet50'.split()
    else:
        methods = 'ResNet50 DDC_ResNet50 KDDE_DDC_ResNet50 CT_DDC_ResNet50'.split()

    n = len(methods) 
    overall_perf_table = np.zeros((n, 3)) 
    
    for i,(source_domain, target_domain) in enumerate(ude_tasks):
        print ('Evaluating [%d] %s->%s' % (i, source_domain, target_domain))
        perf_table = evaluate_per_task(test_collection, source_domain, target_domain, args.run, rootpath)
        for j in range(n):
            assert(perf_table[j][0].startswith(methods[j]))
            overall_perf_table[j,:] += perf_table[j][1:] # source, target, source+target
    overall_perf_table /= len(ude_tasks)
    
    print ('#model source-domain target-domain expanded-domain')
    for j in range(n):
        print ('%s %s' % (methods[j], ' '.join(['%.2f' % x for x in overall_perf_table[j]])))