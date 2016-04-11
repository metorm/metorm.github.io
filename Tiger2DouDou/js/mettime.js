function mettime() {
	var StartTime = new Date('2013/10/06 20:00:00');
	var NowTime = new Date();
	var t = NowTime.getTime() - StartTime.getTime();
	var d=0;
	var h=0;
	var m=0;
	var s=0;
	if(t>=0){
		d=Math.floor(t/1000/60/60/24);
		h=Math.floor(t/1000/60/60%24);
		m=Math.floor(t/1000/60%60);
		s=Math.floor(t/1000%60);
	}

	document.getElementById("day").innerHTML = (d<10?"0"+d:d) + " days";
	document.getElementById("hour").innerHTML = (h<10?"0"+h:h) + " hours";
	document.getElementById("minute").innerHTML = (m<10?"0"+m:m) + " minutes";
	document.getElementById("second").innerHTML = (s<10?"0"+s:s) + " seconds";
}
setInterval(mettime, 200);
