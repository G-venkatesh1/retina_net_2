from pycocotools.cocoeval import COCOeval
import json
import torch
from torchvision.ops import nms

def evaluate_coco(dataset, model, threshold=0.05):
    
    model.eval()
    
    with torch.no_grad():

        # start collecting results
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
                transformed_anchors,classification = model(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
                finalResult = [[], [], []]
                finalScores = torch.Tensor([])
                finalAnchorBoxesIndexes = torch.Tensor([])#.long()
                finalAnchorBoxesCoordinates = torch.Tensor([])

                if torch.cuda.is_available():
                    finalScores = finalScores.cuda()
                    finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                    finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

                for i in range(classification.shape[2]):
                    scores = torch.squeeze(classification[:, :, i])
                    scores_over_thresh = (scores > 0.05)
                    if scores_over_thresh.sum() == 0:
                            # no boxes to NMS, just continue
                        continue

                    scores = scores[scores_over_thresh]
                    anchorBoxes = torch.squeeze(transformed_anchors)
                    anchorBoxes = anchorBoxes[scores_over_thresh]
                    anchors_nms_idx = nms(anchorBoxes, scores, 0.5)
                        # val =torch.tensor(anchors_nms_idx)
                        # val2 = torch.tensor([i] * anchors_nms_idx.shape[0])
                        # print(val)
                        # print("\n")
                        # print(val2)
                        # print("\n")
                    finalResult[0].extend(scores[anchors_nms_idx])
                        # finalResult[1].extend(val)
                    finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                        #print(finalResult[1])
                    finalResult[2].extend(anchorBoxes[anchors_nms_idx])

                        #  finalAnchorBoxesIndexesValue = val.cuda()
                    finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0]).cuda()
                        # if torch.cuda.is_available():
                        #     finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                    finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue)).cuda()
                    finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx])).cuda()
        #         # print("values in mode.py",finalScores,finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates)
                    finalScores = torch.cat((finalScores, scores[anchors_nms_idx])).cuda()
                    scores,labels,boxes = finalScores,finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates
                # print("printing in main",scores,labels,boxes)
            # else:
                # scores, labels, boxes = model(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
                    scores = scores.cpu()
                    labels = labels.cpu()
                    boxes  = boxes.cpu()

                    # correct boxes for image scale
                    boxes /= scale
                    # print(scores.shape,labels.shape,boxes.shape)
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
