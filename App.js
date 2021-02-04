import { StatusBar } from 'expo-status-bar';
import React, { useState } from 'react';
import { Button, Image, StyleSheet, Text, TextInput, View } from 'react-native';
import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';
import { fetch } from '@tensorflow/tfjs-react-native';
import * as jpeg from 'jpeg-js';
import { buffer } from '@tensorflow/tfjs';

export default function App() {
  const [url, setUrl] = useState('https://upload.wikimedia.org/wikipedia/commons/c/c4/Habib_Bourguiba_Portrait.jpg');
  const [phase, setPhase] = useState('loading')
  const [result, setResult] = useState(null)
  const getPrediction = async (url) => {
    setPhase('loading Tenser flow')
    await tf.ready()
    setPhase('loading Mobilenet')
    const model = await mobilenet.load()
    setPhase('fetching image')
    const response = await fetch(url, {}, { isBinary: true })
    const imgData = await response.arrayBuffer();
    setPhase('Getting image tensor')
    const imageTensor = imgToTensor(imgData);
    setPhase('Getting image classification')
    const prediction = await model.classify(imageTensor);
    console.log(prediction)
    setResult(prediction)

  }

  const imgToTensor = (rawdata) => {
    const { width, height, data } = jpeg.decode(rawdata, true);
    const buffer = new Uint8Array(width * height * 3);
    let offset = 0;
    for (let i = 0; i < buffer.length; i += 3) {
      buffer[i] = data[offset] // R
      buffer[i + 1] = data[offset + 1] //G
      buffer[i + 2] = data[offset + 2] //B
      offset += 4
    }
    return tf.tensor3d(buffer, [height, width, 3])
  }

  return (
    <View style={styles.container}>
      <Text>Classify testing</Text>
      <StatusBar style="auto" />
      <TextInput style={{ width: "80%", height: 40, borderColor: 'red', borderWidth: 2 }} onChangeText={(text) => { setUrl(text) }} value={url} ></TextInput>
      <Text>{phase}</Text>
      <Image style={{width:"70%", height:"50%"}} source={{ uri: url }}></Image>
      <Button title="Do it" onPress={() => { getPrediction(url) }} ></Button>
      <Text> {result ? result[0]['className']: null } is Found</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
