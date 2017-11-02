var dist = require('vectors/dist-nd');
var add = require('vectors/add-nd');
var fs = require('fs');

var values = {};
var weightsize = 2;
var biassize = 2;
var sizes = [784,80,30,30,10];

var main=function()
{
	//get a random image from the database.
	values = dataread(Math.floor(Math.random()*60000));

	//sizes is an array of the neural network layer sizes. The first must equal the number of inputs, for example mine is 784 because each image is 
	//28x28px and outputs a single grayscale value, meaning it's 28x28=784 inputs. 
	//the last number must equal the desired output. In mine I chose 10 because I want each value to correspond to the correct output, for example
	//I want the handwritten digit 0 to be represented as [1,0,0,0,0,0,0,0,0,0,0]
	network.init(sizes);
	network.generateInputs();
	network.run();
}


var network=
{
	neurons:[],
	init:function(sizes){
		//initialize function. This means create the neurons on each "level"
		for (var i = 0; i<sizes.length; i++)
		{
			this.neurons[i]=[]
			if (i>0)
			{
				//console.log('Sizes[i]: '+sizes[i])
				for (var ii=0; ii<sizes[i];ii++)
				{
					//"Hidden" or non-first-level neurons. These do all the fun stuff.
					//console.log("Hidden Neuron created.")
					this.neurons[i][ii]=newneuron_hidden();
				}
			}
			else
			{
				//the first "level" is the input neurons that read the handwriting. This is the only part which changes from project to project (the creation doesn't but the output function in createinput does)
				for (var ii=0; ii<sizes[i];ii++)
				this.neurons[0][ii]=newneuron_input(ii)
			}
		}
	},
	generateInputs:function(){
		//generate input links for non-input neurons. Input neurons get their input from the raw data they're fed. 
		for (var i = 1; i<this.neurons.length; i++)
		{
			//ignore i=0 because this level is the input neurons
			for (var ii = 0; ii<this.neurons[i].length;ii++)
			{
				//every neuron gets inputs from ALL of the outputs of the layer above it. So we point the "inputsbasic" list to the previous layer.
				this.neurons[i][ii].inputsbasic=this.neurons[i-1];
				//run the weightbias generation program for this new neuron
				this.neurons[i][ii].init_weightbias();
			}
		}
	},	
	run:function(){
		//run input/output handling for each level.
		for (var i=0; i<this.neurons.length; i++)
		{
			for (var ii=0; ii<this.neurons[i].length; ii++)
			{
				//each neuron layer runs its output function and stores it in its result variable.
				this.neurons[i][ii].result=this.neurons[i][ii].output();
				//console.log("Output at ("+i+","+ii+") is: "+this.neurons[i][ii].result)
			}
		}

		//results of last neurons have been completed. You should be able to read results. Expect noise until it learns. 
		
		outdest="hd"
		console.log('Neurons.length: '+this.neurons.length)
		for (var i = 1; i<this.neurons.length; i++)
		{
			//this just writes the output to a string of <p> elements 
			outdest=".hd"+i;
			//look in index.html for .hd1,2,3 etc
			outstr=""
			//console.log('Neurons[i].length: '+this.neurons[i].length)
			for (ii in this.neurons[i])
			{
				outstr+="<p>"+this.neurons[i][ii].result+"</p>";
			}
			if (i==1)
			$('.hd1').html(outstr);

			if (i==2)
			$('.hd2').html(outstr);

			if (i==3)
			$('.hd3').html(outstr);

			if (i==4)
			$('.hd4').html(outstr);
		}

	}
}


function sigmoid(z)
{
	return 1/(1+Math.exp(-z));
}

function newneuron_hidden()
{
	var n={
		init_weightbias:function(){
			//generates this neuron's weight/bias list
			//console.log("Input size: "+this.inputsbasic.length)
			for (var i=0; i<this.inputsbasic.length; i++)
			{
				//console.log("input added to list!")
				//for each input we need to generate a random weight and bias. These values, while random, form the basis of how it 'Thinks'
				//target is the raw source of the input
				this.inputs[i]={
					target:this.inputsbasic[i],
					weight:(Math.random()*weightsize)-(weightsize/2),
					bias:(Math.random()*biassize)-(biassize/2)
				}
			}
		},
		inputs: [],
		inputsbasic:[],
		s: function(inputs){
			var sum = 0;
			//console.log(this.inputs.length)
			for (var i=0; i<this.inputs.length; i++)
			{
				//so this is probably the most complicated bit.
				//what you're doing is summing all the input values we have then doing the weight/bias bits to each.  
				sum+=(this.inputs[i].target.result*this.inputs[i].weight)+this.inputs[i].bias;
			}
			//once we have this value's sum, we throw it into the sigmoid, which "smooths" out an answer instead of raw thresholds.
			//I don't know why this is necessary.
			return sigmoid(sum);
		},
		output:function(){
			var r=this.s(this.inputs)
			return r;
		},
		result: 0
	}
	return n
}

function newneuron_input(xy){
	//This is an input. It only has a position and an output function ¯\_(ツ)_/¯
	var n = {
		xy:xy,
		output:function(){
			//this is the only bit which changes from NN to NN. This interprits the raw data. Can be anything as long as it's a list of integers. 
			//in this case it's just grayscale values of handwritten digits. 
			return values.pixels[this.xy];
		},
		result:0
	}

	return n;
}

function dataread(image)
{
	//this reads the NIST data. If you're using my raw code, do this however you like. 
	var dataFileBuffer  = fs.readFileSync(__dirname + '/train-images-idx3-ubyte/train-images.idx3-ubyte');
	var labelFileBuffer = fs.readFileSync(__dirname + '/train-labels-idx1-ubyte/train-labels.idx1-ubyte');
	var pixelValues     = [];
 
	var pixels = [];

	for (var x = 0; x <= 27; x++) {
		for (var y = 0; y <= 27; y++) {
			pixels.push(dataFileBuffer[(image * 28 * 28) + (x + (y * 28)) + 15]);
		}
	}

    var imageData  = {};
    imageData.label 	= JSON.stringify(labelFileBuffer[image + 8])
    imageData.pixels 	= pixels;

	return imageData;
}

function successvector(label){
	//this also changes from project to project. Basically what we want a true result to look like.
	var r = [];
	if (label==0)
	{
		r=[1,0,0,0,0,0,0,0,0,0]
	}

	if (label==1)
	{
		r=[0,1,0,0,0,0,0,0,0,0]
	}

	if (label==2)
	{
		r=[0,0,1,0,0,0,0,0,0,0]
	}

	if (label==3)
	{
		r=[0,0,0,1,0,0,0,0,0,0]
	}

	if (label==4)
	{
		r=[0,0,0,0,1,0,0,0,0,0]
	}

	if (label==5)
	{
		r=[0,0,0,0,0,1,0,0,0,0]
	}

	if (label==6)
	{
		r=[0,0,0,0,0,0,1,0,0,0]
	}

	if (label==7)
	{
		r=[0,0,0,0,1,0,0,1,0,0]
	}

	if (label==8)
	{
		r=[0,0,0,0,1,0,0,0,1,0]
	}

	if (label==9)
	{
		r=[0,0,0,0,0,0,0,0,0,1]
	}

	return r;
}

$(document).ready(main)