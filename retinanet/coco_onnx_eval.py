from pycocotools.cocoeval import COCOeval
import json
import torch
import onnx
import onnxruntime

def evaluate_coco_onnx(dataset, ort_session, threshold=0.05):
    ort_inputs = {'input.1': None}
    with torch.no_grad():
        results = []
        image_ids = []
        c=0
        for index in range(len(dataset)):
            
            data = dataset[index]
            scale = data['scale']
            if(c>15): break
            c=c+1
            # run network
            if torch.cuda.is_available():
                inputs = data['img'].permute(2, 0, 1).float().unsqueeze(dim=0)
                ort_inputs['input.1'] = inputs.cpu().numpy()
                ort_outs = ort_session.run(None, ort_inputs)
                scores, labels, boxes = ort_outs[0], ort_outs[1], ort_outs[2]
            # else:
            #     scores, labels, boxes = model(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
            # scores = scores.cpu()
            # labels = labels.cpu()
            # boxes  = boxes.cpu()

            # correct boxes for image scale
            # boxes /= scale
            print(scores.shape,labels.shape,boxes.shape)
            if boxes.shape[0] > 0:
                # change to (x, y, w, h) (MS COCO standard)
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]

                # compute predicted labels and scores
                #for box, score, label in zip(boxes[0], scores[0], labels[0]):
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    # scores are sorted, so we can break
                    if score < threshold:
                        break

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id'    : dataset.image_ids[index],
                        'category_id' : dataset.label_to_coco_label(label),
                        'score'       : float(score),
                        'bbox'        : box.tolist(),
                    }

                    # append detection to results
                    results.append(image_result)

            # append image to list of processed images
            image_ids.append(dataset.image_ids[index])

            # print progress
            print('{}/{}'.format(index, len(dataset)), end='\r')

        if not len(results):
            return

        # write output
        json.dump(results, open('{}_bbox_results.json'.format(dataset.set_name), 'w'), indent=4)

        # load results in COCO evaluation tool
        coco_true = dataset.coco
        coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(dataset.set_name))

        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # model.train()

        return
