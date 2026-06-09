import { makeShapeDataset, rasterizeShape, extractShapeFeatures, featuresToVec, standardize, applyStandardize, trainTestSplit, KNN, SHAPE_LABELS } from './src/lib/ml';
const W=280,H=280;
const ds = makeShapeDataset(800, 11, W, H);
const {X:Xs, mean, std} = standardize(ds.X);
const {XTrain,yTrain,XTest,yTest} = trainTestSplit(Xs, ds.y, 0.2);
const m = new KNN(7); m.fit(XTrain,yTrain,SHAPE_LABELS);
console.log('test acc', m.accuracy(XTest,yTest));

// simulate user drawing circle - rasterize with thick brush
function rng(seed){let s=seed;return()=>{s=Math.imul(s^s>>>15,s|1);s^=s+Math.imul(s^s>>>7,s|61);return((s^s>>>14)>>>0)/4294967296}}
for(let cls=0; cls<4; cls++){
  let correct=0;
  for(let i=0;i<20;i++){
    const r=rng(1000+cls*100+i);
    const mask = rasterizeShape(cls, W, H, r, 10);
    const f = extractShapeFeatures(mask,W,H);
    const p = m.predict(applyStandardize(featuresToVec(f), mean, std));
    if(p.label===SHAPE_LABELS[cls]) correct++;
  }
  console.log(SHAPE_LABELS[cls], correct+'/20');
}
