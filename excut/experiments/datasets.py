import os


datasets={}
experiment_name="expr10"

# add baseline_data
for ds in ['terroristAttack', 'imdb', 'uwcse', 'webkb', 'mutagenesis', 'hep']:
    dataset_folder = os.path.join('/scratch/GW/pool0/gadelrab/ExDEC/data/baseline_data/', ds)
    dataset_output_folder = os.path.join('/scratch/GW/pool0/gadelrab/ExDEC/%s/baseline_data/' % experiment_name, ds)
    datasets[ds] = {'dataset_folder': dataset_folder,
                    'dataset_output_folder': dataset_output_folder,
                    'target_entities': os.path.join(dataset_folder, '%s_target_entities' % ds),
                    'kg': os.path.join(dataset_folder, '%s_kg' % ds),
                    'kg_idntifier': 'http://%s_kg.org' % ds,
                    'data_prefix': 'http://exp-data.org'
                    }

# Add yago related data
for ds in ['yago_art_3_4k']: #, 'yago_art_3_filtered_target', 'yago_art_3_4k']:
    dataset_folder = '/scratch/GW/pool0/gadelrab/ExDEC/data/yago/'
    dataset_output_folder = os.path.join('/scratch/GW/pool0/gadelrab/ExDEC/%s/yago/' % experiment_name, ds)
    datasets[ds] = {'dataset_folder': dataset_folder,
                    'dataset_output_folder': dataset_output_folder,
                    'target_entities': os.path.join(dataset_folder, '%s.tsv' % ds),
                    'kg': '/scratch/GW/pool0/gadelrab/ExDEC/data/yago/yagoFacts_3.tsv',
                    'kg_idntifier': 'http://yago-expr.org',
                    'data_prefix': 'http://exp-data.org',
                    'safe_url': True
                    }

for ds in ['grad_ungrad_course']:
    dataset_folder = '/scratch/GW/pool0/gadelrab/ExDEC/data/uobm/'
    dataset_output_folder = os.path.join('/scratch/GW/pool0/gadelrab/ExDEC/%s/uobm/' % experiment_name, ds)
    datasets[ds] = {'dataset_folder': dataset_folder,
                    'dataset_output_folder': dataset_output_folder,
                    'target_entities': os.path.join(dataset_folder, '%s.ttl' % ds),
                    'kg': '/scratch/GW/pool0/gadelrab/ExDEC/data/uobm/uobm10_kg.nt',
                    'kg_idntifier': 'http://uobm10.org'
                    }
