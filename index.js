import fs from 'fs'
import * as tf from '@tensorflow/tfjs-node'
import * as mobilenet from '@tensorflow-models/mobilenet'

const imagecat = `${__dirname}/assets/cat.jpg`
const imagedog = `${__dirname}/assets/golden-retriever.jpg`

const image = fs.readFileSync(imagecat)
const decodedimage = tf.node.decodeImage(image, 3)

async function App(){

    const model = await mobilenet.load()
    const predictions = model.classify(decodedimage)
    console.log(`Predictions: ${JSON.stringfy(predictions, undefined, 2)}`);

}

App()