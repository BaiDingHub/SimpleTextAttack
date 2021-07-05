import datasets



# dataset = datasets.load_dataset('ag_news', None)['train']
datasets_list  = datasets.list_datasets()
print(', '.join(dataset for dataset in datasets_list))