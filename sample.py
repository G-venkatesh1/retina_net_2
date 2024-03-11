import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, Resizer, Normalizer
from retinanet import coco_eval
import onnx
def main(args=None):
    model_path ='/kaggle/input/retina_net/pytorch/model/1/coco_resnet_50_map_0_335_state_dict.pt'
    retinanet = model.resnet50(num_classes=80, pretrained=True).cuda()
    retinanet.load_state_dict(torch.load(model_path))
    example_input = torch.randn(1, 3,640,640).cuda()
    onnx_path = "ret.onnx"
    torch.onnx.export(retinanet,example_input,onnx_path,opset_version=13)                              # model being run
    
    
    



if __name__ == '__main__':
    main()
