'use strict'
var dist = require('vectors/dist-nd');
var add = require('vectors/add-nd');
var fs = require('fs');

var values = {};
var weightsize = 10;
var biassize = 10;

var canvas = {}
var context = {}

var runlist = [];
var imagelist = [];

var main=function(){

	preread();

	canvas = document.getElementById("canvas");
	context = canvas.getContext("2d");

	//get a random image from the database.
	values = dataread(Math.floor(Math.random()*60000));

	$('.Plzno').click(function(event) {
		localStorage.removeItem("Handwriting_NN_lastrun")
		network.init();
		network.generateInputs();
		network.run();
	});

	$('.gobutton').click(function(event) {
		//for the github version let's just do this once. And never change variables.
		var runcount=16;
		for (var i=0; i<runcount; i++)
		{
			//get a random image from the database.
			console.log("starting dataread");
			values = dataread(Math.floor(Math.random()*60000));
			console.log("ending dataread");
			runEpoch();

			console.log("Testing #:"+values.label+", Global progress: "+((Math.round(((i*100)/runcount))*100)/100)+"%");
			drawCanvas();
		}
	});

	$('.1rungo').click(function(event) {
		//for the github version let's just do this once. And never change variables.
		for (var i=0; i<1; i++)
		{
			//get a random image from the database.
			console.log("starting dataread");
			values = dataread(Math.floor(Math.random()*60000));
			console.log("ending dataread");
			runEpoch();

			console.log("Testing #:"+values.label);
			drawCanvas();
		}
	});

	network.init(sizes);
	network.generateInputs();

	//retreive last run and restore it.
	if (localStorage.getItem("Handwriting_NN_lastrun")!==null)
	{
		var cor=JSON.parse(localStorage.getItem("Handwriting_NN_lastrun"))
		network.pushCoordsToNetwork(cor.coords)
		console.log(localStorage.getItem("Handwriting_NN_lastrun"))
	}
	else
	{
		alert("this looks like your first run.")
	}

	network.run();
	runlist.push({cost:network.getCost(),label:values.label})
	drawCanvas();
	$(".run1here").text(network.getCost())
}
//sizes is an array of the neural network layer sizes. The first must equal the number of inputs, for example mine is 784 because each image is 
//28x28px and outputs a single grayscale value, meaning it's 28x28=784 inputs. 
//the last number must equal the desired output. In mine I chose 10 because I want each value to correspond to the correct output, for example
//I want the handwritten digit 0 to be represented as [1,0,0,0,0,0,0,0,0,0,0]

var sizes = [784,20,10];

var runEpoch=function()
{
	console.clear();
	//now the climber has put his foot in a random place on the terrain. We need to store the height of his foot. 
	console.log("starting store");
	network.store();
	console.log("Store finished")
	//next, the climber must put his other foot around the terrain near his current step, and feel around until he has a better spot.

	var origcoords = network.historicalweightbias[network.historicalweightbias.length-1].coords;
	console.log("starting makenewbiases");
	var delts=network.makeNewBiases();
	//ok so we assume he knows the direction to put his next foot.
	var line_samples=[];
	var maxdis=10;
	var curdis=0;

	//ok so originally, I would have been satisfied at "delts" and applied them to the network. but unforuntately
	//delts is pretty much an arbitrary distance (the direction is right, but I digress)
	//so what we'll need to do is do a straight-line minimization step.

	//so now imagine the climber has walked in a straight line for an arbitrary (and long) distance, specified by maxdis.
	//he now remembers the lowest point, and that's his next step.
	for (var i=0; i<100; i++)
	{
		network.pushCoordsToNetwork(origcoords)
		network.applydelts(delts,i/(100/maxdis));
		network.run();
		var c = network.getCost()
		line_samples.push({scalar:i/(100/maxdis),cost:c})
		console.log("Testing scalar "+i/(100/maxdis)+", cost was: "+c)
	}

	//now we have a list of points on this line. Let's find the minimum and go from there.
	var curmin=1000000000;
	var minid=0;
	for (var i=0; i<line_samples.length; i++)
	{
		if (line_samples[i].cost<curmin)
		{
			curmin=line_samples[i].cost;
			minid=i;
		}
	}
	console.log("Best scalar found at "+line_samples[minid].scalar+" With value "+line_samples[minid].cost)

	//now we have the lowest point on that line. That'll be our next start.
	network.pushCoordsToNetwork(origcoords);
	network.applydelts(delts,line_samples[minid].scalar)
	network.run();
	//we should now have the minimum point for this line. or at least a much larger line than it was. 

	$(".run2here").text(network.getCost())
	runlist.push({cost:network.getCost(),label:values.label})
	
}


var network=
{
	neurons:[],
	learnSpeedLimiter: 10,
	historicalweightbias:[],
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
		
		var outdest="hd"
		//console.log('Neurons.length: '+this.neurons.length)
		for (var i = 1; i<this.neurons.length; i++)
		{
			//this just writes the output to a string of <p> elements 
			outdest=".hd"+i;
			//look in index.html for .hd1,2,3 etc
			var outstr=""
			//console.log('Neurons[i].length: '+this.neurons[i].length)
			for (ii in this.neurons[i])
			{
				outstr+="<p>"+this.neurons[i][ii].result+"</p>";
			}

			if (i==this.neurons.length-1)
			$('.hd4').html(outstr);
		}
	},
	getCost:function(){
		var opts=[];
		for (var i in this.neurons[this.neurons.length-1])
		{
			//Basically this makes the outputs a lot easier to read by putting them into a variable.
			opts[i]=this.neurons[this.neurons.length-1][i].result
		}


		return disttosuccess(values.label,opts)
	},
	store:function(){
		//ok so with the run that we just made, we should have a list of weights and biases for each neuron. 
		//we need to store all that first, so that future runs can use it. 
		var r = {};

		//"Cost" is kinda deceptive. It really means "success value". The lower the "Cost" is the closer we are to winning.
		//however "Cost" is the standard name to describe this, so here we're stuck.

		r.cost=this.getCost();

		//now we need to encode the values that got us here. 
		r.coords=[];

		r.value=values.label;
		//really, this means that this is a spot in an n-dimensional coordinate plane. I know that sounds stupid. But what we're doing is making
		//a hill-climbing algorithm but instead of a 2D terrain with a 3rd dimension representing height, it's a bajillion dimension thing with a bazzilion+1'th dimension representing cost.
		//we're going to make a machine that can find its way around this "Hill" using the same techniques.
		for (var i=1; i<this.neurons.length; i++)
		{
			for (var ii=0; ii<this.neurons[i].length; ii++)
			{
				//ok, we're at "each neuron" level.
				for (var iii=0; iii<this.neurons[i][ii].inputs.length; iii++)
				{
					//push both the weight and the bias into this list.
					r.coords.push(this.neurons[i][ii].inputs[iii].weight)
					r.coords.push(this.neurons[i][ii].inputs[iii].bias)
				}

			}
		}
		console.log("The network is going to run a total of "+r.coords.length+" times")
		//now we should have every single possible change to make in "coordinates" stored in the coords property. Whew. 
		this.historicalweightbias.push(r);
		//we now have the "Foot position" of this climber. 
		//to avoid using "bazillion" we're calling the current r.coords.length "Z". It should number in the hundreds of thousands. 
		//actual calculations show that it's 132,600 (Or something around there) for this NN. 

		//and now to ensure we don't lose all of our progress at every restart, the arguments will now be stored in localstorage.
		localStorage.setItem("Handwriting_NN_lastrun",JSON.stringify(r))
	},
	makeNewBiases:function(){
		//delt[coordval] represents our best guess of the correct direction to go with that particular variable. 
		var delt=[];

		//rr=relevant record
		var rr=this.historicalweightbias[this.historicalweightbias.length-1]

		var ltot=rr.coords.length;

		for (var i=0; i<rr.coords.length; i++)
		{
			var newcoords=rr.coords.slice();
			//for each coordinate, we need to get the current value, add a random value (positive or negative) then feed it back into the network.

			// how much we changed this value
			var deltaval=(Math.random()*weightsize/2)-(weightsize/4);

			if (Math.abs(deltaval)<1)
			deltaval = Math.sign(deltaval);

			newcoords[i]+=deltaval;
			//now we need to feed it back into the network
			this.pushCoordsToNetwork(newcoords)

			this.run();

			var newCost=this.getCost()

			//now we have a change in cost that is associated with this particular changed value.
			//let's say that the cost is now higher.
			//this indicates this particular coordinate is in the wrong direction. We should ensure that delt[thisval] is in the opposite direction.
			//so the "old" coordinate was 1, and the "new" coordinate is 2, making our deltacoord 1. Our old cost was 3, and now our cost is 4.
			//making our deltacost 1. Thus delt[thisval] should be negative. 
			var deltacost=rr.cost-newCost

			//slope is rise over run kids. So what we've done here is established a direction to make the thing go. It's pointing in the direction the next
			//step should take. 
			delt[i]=(deltacost/deltaval);

			console.log("Individual Progress: "+i+"/"+ltot+" or "+((Math.round(((i*100)/ltot))*100)/100)+"%")
		}

		//after all of these runs (again 1 per coord, "Z")
		return delt;
	},
	pushCoordsToNetwork:function(coords){
		var v=0;
		//V represents the postion in the coords array. 
		for (var i=1; i<this.neurons.length; i++)
		{
			for (var ii=0; ii<this.neurons[i].length; ii++)
			{
				//ok, we're at "each neuron" level.
				for (var iii=0; iii<this.neurons[i][ii].inputs.length; iii++)
				{
					this.neurons[i][ii].inputs[iii].weight=coords[v]
					v+=1;
					this.neurons[i][ii].inputs[iii].bias=coords[v]
					v+=1;
				}
			}
		}
	},
	applydelts:function(delts,lsl){
		var v=0;
		for (var i=1; i<this.neurons.length; i++)
		{
			for (var ii=0; ii<this.neurons[i].length; ii++)
			{
				//ok, we're at "each neuron" level.
				for (var iii=0; iii<this.neurons[i][ii].inputs.length; iii++)
				{
					this.neurons[i][ii].inputs[iii].weight+=delts[v]*lsl;
					v+=1;
					this.neurons[i][ii].inputs[iii].bias+=delts[v]*lsl;
					v+=1;
				}
			}
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

function preread(){
		//this reads the NIST data. If you're using my raw code, do this however you like. 
	var dataFileBuffer  = fs.readFileSync(__dirname + '/train-images-idx3-ubyte/train-images.idx3-ubyte');
	var labelFileBuffer = fs.readFileSync(__dirname + '/train-labels-idx1-ubyte/train-labels.idx1-ubyte');
	

	for (var image = 0; image<59900; image++)
	{
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

    imagelist.push(imageData);
	}

}

function dataread(image)
{
	var imageData = imagelist[image];

	return imageData;
}

function disttosuccess(label,outputs){
	var difvec=[]
	var sv=successvector(label)
	for (var i in outputs)
	{
		difvec[i]=outputs[i]-sv[i]
	}

	var sum=0;
	//difvec is now a vector of the differences between expected and true. We need to turn this into a single value. Here comes distance.
	for (var i in outputs)
	{
		//we're creating a distance. You do this by squaring the components of the vector. For example distance between (0,0) and (2,3) is sqrt(2^2+3^2).
		//this holds to n dimensions. However the sqrt is only useful for finding the physical distance, we don't need to bother with that
		//and it'll slow everything down so it's dispensed with. 
		sum+=Math.pow(difvec[i],2);
	}

	//the lower sum is, the better off we are.
	return sum;
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
		r=[0,0,0,0,0,0,0,1,0,0]
	}

	if (label==8)
	{
		r=[0,0,0,0,0,0,0,0,1,0]
	}

	if (label==9)
	{
		r=[0,0,0,0,0,0,0,0,0,1]
	}

	return r;
}

function drawCanvas(){

	context.clearRect(0,0,canvas.width,canvas.height);

	var steps = (canvas.width-60)/runlist.length;

	//draw horizontal axis
	context.beginPath();
	context.moveTo(30,canvas.height-60)
	context.lineTo(canvas.width-30,canvas.height-60);
	context.strokeStyle="black";
	context.lineWidth=1;
	context.lineJoin="round";
	context.stroke();

	context.fillText("Horizontal axis: Epoches (major runs). The numbers indicate what digit being tested. Vertical axis: Cost. Lower is better.",30,canvas.height-15)


	if (runlist.length>0)
	{
		//make axis points
		for (var i in runlist)
		{
			//only draw certain labels. Don't want to get it too cluttered.
			if ((runlist.length<15) || (runlist.length<30 && runlist.length>15 && ((i%4)==1)) || (runlist.length<600 && runlist.length>30 && ((i%8)==1))|| i==(runlist.length-1))
			{
				context.beginPath();
				context.strokeStyle="gray";
	
				context.moveTo(30+(i*steps),canvas.height-45)
				context.lineTo(30+(i*steps),(canvas.height-45)-(runlist[i].cost*(540/7)))
				context.fillText(runlist[i].label,30+(i*steps),canvas.height-30)
				context.fillText("Cost: "+(Math.round(runlist[i].cost*1000)/1000),30+(i*(steps)),(canvas.height-60)-(runlist[i].cost*(540/7))-15)
				context.stroke();
			}
			else
			{
				//however zero cost points are always significant, so always label them
				if (runlist[i].cost<.01)
				{
					context.fillText(runlist[i].label,30+(i*steps),canvas.height-30)
				}
			}
		}
	
		context.beginPath()
		
		context.moveTo(30,(canvas.height-60)-(runlist[0].cost*(540/7)))
		for (var i in runlist)
		{
			context.lineTo(30+(i*(steps)),(canvas.height-60)-(runlist[i].cost*(540/7)))
		}
		context.lineWidth=4;
		context.lineCap="round";
		context.strokeStyle="blue";
		context.stroke();
	}
}

$(document).ready(main)