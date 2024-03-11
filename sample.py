import torch
from torchvision import transforms
import onnxruntime
from retinanet import model
from retinanet.dataloader import CocoDataset, Resizer, Normalizer,Resizer_const
from retinanet import coco_eval
import onnx
def main(args=None):
    model_path ='/kaggle/input/retina_net/pytorch/model/1/coco_resnet_50_map_0_335_state_dict.pt'
    retinanet = model.resnet50(num_classes=80, pretrained=True).cuda()
    retinanet.load_state_dict(torch.load(model_path))
    example_input = torch.randn(1, 3,608,1024).cuda()
    onnx_path = "ret2.onnx"
    torch.onnx.export(retinanet,example_input,onnx_path,opset_version=13)                              # model being run
    ort_session = onnxruntime.InferenceSession('/kaggle/working/retina_net_2/ret2.onnx')
    ort_inputs = {' input.1': None}
    dataset_val = CocoDataset('/kaggle/input/coco-2017-dataset/coco2017', set_name='val2017',
                              transform=transforms.Compose([Normalizer(), Resizer()])) 
    data = dataset_val[0]
    inputs = data['img'].permute(2, 0, 1).float().unsqueeze(dim=0)
    ort_inputs['input.1'] = inputs.cpu().numpy()
    ort_outs = ort_session.run(None, ort_inputs)
    scores, labels, boxes = ort_outs[0], ort_outs[1], ort_outs[2] 
    print(scores.shape,labels.shape,boxes.shape)   
    
    



if __name__ == '__main__':
    main()
