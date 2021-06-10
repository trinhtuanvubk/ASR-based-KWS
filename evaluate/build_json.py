def build_json(data_dir):    
    # args : data_dir 
    # vocabulary
    # train_dir = os.path.join(args.data_dir, 'train')
    # dev_dir = os.path.join(data_dir, 'dev')
    dev_dir = os.path.join(data_dir,"data")
    vocab = os.listdir(data_dir)
    print(vocab)
    # vocab.remove('_background_noise_')
    # vocab = sorted(vocab)[:args.n_keyword]
    n_keyword = 11
    vocab = ['_unknown_', 'down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
    print(vocab)
   
    # with open(f'{data_dir}/dev_{n_keyword}.json', 'w') as f:
    with open('./_dev_11.json', 'w') as f:
        # keyword = {'file' : [],'text' : word,'label' : y}
        for y, word in enumerate(vocab):
            line = {'file' : [],'text' : word,'label' : y}
            if word == '_unknown_':
                unk_word = [w for w in os.listdir(dev_dir) if w not in vocab]
                for unk in unk_word:
                    files = os.listdir(os.path.join(dev_dir, unk))
                    random.shuffle(files)
                    files = files[:20]
                    for file in files:
                        path =  os.path.join(os.path.join(dev_dir, unk), file)
                        line = {'file': path, 'text': word, 'label': y}
                        # line['file'].append(path)
                        json.dump(line, f)
                        f.write('\n')
            else:
                files = os.listdir(os.path.join(dev_dir, word))
                # print(word, len(files))
                for file in files[:100]:
                    path =  os.path.join(os.path.join(dev_dir, word), file)
                    line = {'file': path, 'text': word, 'label': y}
                    # line['file'].append(path)
                    json.dump(line, f)
                    f.write('\n')

