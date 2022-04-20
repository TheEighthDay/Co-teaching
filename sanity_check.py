import os 
from common import ROOT_PATH,col2anno,read_concept_list,get_anno_path,\
read_anno,read_full_anno,read_domain_list,read_imset,get_pred_path,read_imset,read_pred

def verify_anno(collection, domain, rootpath=ROOT_PATH):
    annotation_name = col2anno[collection]
    concepts = read_concept_list(collection, annotation_name, rootpath)
    pos_set = []
    for concept in concepts:
        anno_path = get_anno_path(collection, domain, annotation_name, concept, rootpath)
        pos_img_list = read_anno(anno_path)
        pos_set += pos_img_list

    # since it is a multi-class problem, there shall be overlap between positive sets of distinct concepts
    assert(len(set(pos_set)) == len(pos_set)) 
    whole_set = read_imset(collection, domain, rootpath)
    common = set(pos_set).intersection(set(whole_set))
    assert(len(common) == len(pos_set))
    assert(len(common) == len(whole_set))
    

def verify_imsets(collection, rootpath=ROOT_PATH):
    domain_list = read_domain_list(collection, rootpath)
    subset = []
    for domain in domain_list:
        imset_per_domain = read_imset(collection, domain, rootpath)
        print ('%s, %s, %d images' % (collection, domain, len(imset_per_domain)))
        subset += imset_per_domain

    whole_set = read_imset(collection, collection, rootpath)
    assert(len(subset) == len(whole_set))
    common = set(subset).intersection(set(whole_set))
    assert (len(common) == len(whole_set))
    print('%s -> %d images' % (collection, len(whole_set)))


def verify_pred(collection, domain, pred_path, rootpath=ROOT_PATH):
    #print ('verifying (%s,%s) -> %s' % (collection, domain, pred_path))
    im2concept = read_pred(pred_path)
    testset = read_imset(collection, domain, rootpath)
    assert(len(im2concept) == len(testset))
    common = set(im2concept.keys()).intersection(set(testset))
    assert(len(common) == len(testset))

def verify_datasets():
    print("start check datasets")
    assert(os.path.exists("datasets"))
    assert(os.path.exists("datasets/OfficeHome"))
    assert(os.path.exists("datasets/domainnet"))
    officehome_domains = read_domain_list('officehome_test', rootpath=ROOT_PATH)
    domainnet_domains = read_domain_list('domainnet_test', rootpath=ROOT_PATH)

    for domain in officehome_domains:
        assert(os.path.exists(os.path.join("datasets","OfficeHome",domain)))

    for domain in domainnet_domains:
        assert(os.path.exists(os.path.join("datasets","domainnet",domain)))
    print("over")

def verify_collections(collection_list):
    networks = 'DDC_ResNet50 KDDE_DDC_ResNet50'.split()
    run="1"
    
    test_collection = collection_list[1]
    domains = read_domain_list(test_collection, rootpath=ROOT_PATH)
    print ('Domains of %s: %s' % (test_collection, domains))
    print("start check anno & imsets")
    for collection in collection_list:
        for domain in domains:
            verify_anno(collection, domain, ROOT_PATH)
        verify_imsets(collection, ROOT_PATH)
    print("ok, over")


    print("start check pred")
    
    for target_domain in domains:
        pred_path = get_pred_path(test_collection, target_domain, 'ResNet50_%s'%target_domain, run, rootpath=ROOT_PATH)
        verify_pred(test_collection, target_domain, pred_path, rootpath=ROOT_PATH)

        source_domains = list(domains)
        source_domains.remove(target_domain)
        for network in networks:
            for src_domain in source_domains:
                model_name = '_'.join([network, src_domain, target_domain])
                pred_path=get_pred_path(test_collection, target_domain, model_name, run, rootpath=ROOT_PATH)
                verify_pred(test_collection, target_domain, pred_path, rootpath=ROOT_PATH)

    print("ok, over")

if __name__ == '__main__':
    verify_collections(['officehome_train', 'officehome_test'])
    verify_collections(['domainnet_train', 'domainnet_test'])
    verify_datasets()