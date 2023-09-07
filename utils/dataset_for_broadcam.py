from utils.tools_for_broadcam import *

class VOC_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, domain, transforms):
        self.root_dir = root_dir

        self.image_dir = self.root_dir + 'JPEGImages/'
        self.mask_dir = self.root_dir + 'SegmentationClassAug/'
        
        self.image_id_list = open('./data/%s.txt'%domain).readlines()
        
        self.transforms = transforms
        
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
        
        self.normalize_fn = Normalize(mean_vals, std_vals)
        
        data = read_json('./data/VOC_2012.json')
        self.class_dic = data['class_dic']
        self.classes = data['classes']
        
    def __len__(self):
        return len(self.image_id_list)

    def get_image(self, image_id):
        image_id = image_id.split('.')[0].split('/')[-1]
        image = Image.open(self.image_dir + image_id + '.jpg').convert('RGB')
        return image
    
    def __getitem__(self, index):
        line = self.image_id_list[index]
        tags = line.split('\n')[0].split(' ')[0].split('/')[-1]
        image_id = tags.split('.')[0]
        label_list = torch.zeros((20))
        # _, tags = read_xml(self.root_dir + 'Annotations/' + image_id + '.xml')
        tags = np.unique(np.array(Image.open(self.root_dir+'SegmentationClassAug/'+image_id+'.png')))
        for tag in tags:
            if (tag == 0) or (tag==255) :
                continue
            label_list[tag-1] = 1
        img = self.get_image(image_id)
        orisize = img.size
        img = self.normalize_fn(img)
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img)
        
        data_list = img
        
        return data_list, label_list, image_id

class COCO_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, domain, transforms):
        self.root_dir = root_dir

        self.image_dir = self.root_dir + 'JPEGImages/'
        self.mask_dir = self.root_dir + 'SegmentationClass/'
        
        self.image_id_list = open(self.root_dir+'/%s_cls.txt'%domain).readlines()
        
        self.transforms = transforms
        
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
        
        self.normalize_fn = Normalize(mean_vals, std_vals)


    def __len__(self):
        return len(self.image_id_list)

    def get_image(self, image_id):
        image = Image.open(self.image_dir + image_id + '.jpg').convert('RGB')
        return image
    
    def __getitem__(self, index):
        line = self.image_id_list[index]
        tags = line.split('\n')[0].split(' ')
        image_id = tags[0]
        label_list = torch.zeros((80))
        for i in range(1, len(tags)):
            label_list[int(tags[i])] = 1
        img = self.get_image(image_id)
        img = self.normalize_fn(img)
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img)
        
        data_list = img
        
        return data_list, label_list, image_id

class TrainDataset(Dataset):
    def __init__(self, data_path, transform=None, dataset_name=None, scales = [1]):
        self.data_path = data_path
        self.transform = transform
        self.dataset_name = dataset_name
        self.object = self.path_label()
        self.scales = scales

    def __getitem__(self, index):
        fn, label = self.object[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.object)
        
    def path_label(self):
        path_label = []
        if (self.dataset_name == 'luad') or (self.dataset_name == 'bcss'):
            for root, dirname, filenames in os.walk(self.data_path):
                for f in filenames:
                    image_path = os.path.join(root, f)
                    fname = f[:-4]
                    ##  Extract the image-level label from the filename
                    ##  LUAD-HistoSeg   : 'patient_ID'+'_x-axis'+'_y-axis'+'[a b c d]'.png
                    ##  BCSS-WSSS       : 'Image-name-of-BCSS'+'+index'+'[abcd]'.png
                    label_str = fname.split(']')[0].split('[')[-1]
                    if self.dataset_name == 'luad':
                        image_label = torch.Tensor([int(label_str[0]),int(label_str[2]),int(label_str[4]),int(label_str[6])])
                    elif self.dataset_name == 'bcss':
                        image_label = torch.Tensor([int(label_str[0]),int(label_str[1]),int(label_str[2]),int(label_str[3])])
                    path_label.append((image_path, image_label))
        return path_label