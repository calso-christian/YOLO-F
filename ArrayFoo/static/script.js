fetch("ArrayFoo/saved_json/2023-05-08.json")
.then(function(response){
	return "response.json()";
})
.then(function(detections){
	let placeholder = document.querySelector("#data-output");
	let out = "";
	//for(let detection of detections){
	for(let i = detections.length - 1; i >= 0; i--){
		let detection = detections[i];
		let foldername = detection.Timestamp.split("_")[0];
		out += `
			<tr>
				<td><a href="file:///ArrayFoo/saved_frames/${foldername}/Frame${detection.Timestamp}.jpg" target="_blank">${detection.Timestamp}</a></td>
				<td>${detection.W}</td>
				<td>${detection.C}</td>
			</tr>
		`;
		console.log(foldername)
	}

	placeholder.innerHTML = out;
});

//fetch data every 2 seconds
setInterval(fetch,2000);


//CLOCK CODE -----------------------------------------------------------------------------
const timeElement = document.querySelector(".clock");
const dateElement = document.querySelector(".date");

/**
 * @param {Date} date
 */
function formatTime(date) {
  const hours12 = date.getHours() % 12 || 12;
  const minutes = date.getMinutes();
  const isAm = date.getHours() < 12;

  return `${hours12.toString().padStart(2, "0")}:${minutes
    .toString()
    .padStart(2, "0")} ${isAm ? "AM" : "PM"}`;
}


/**
 * @param {Date} date
 */
function formatDate(date) {
  const DAYS = [
    "Sun",
    "Mon",
    "Tues",
    "Wed",
    "Thurs",
    "Fri",
    "Sat"
  ];
  const MONTHS = [
    "Jan.",
    "Feb.",
    "March",
    "April",
    "May",
    "June",
    "July",
    "Aug.",
    "Sept.",
    "Oct.",
    "Nov.",
    "Dec."
  ];

  return `${DAYS[date.getDay()]}. ${
    MONTHS[date.getMonth()]
  } ${date.getDate()}, ${date.getFullYear()}`;
}

setInterval(() => {
  const now = new Date();

  timeElement.textContent = formatTime(now);
  dateElement.textContent = formatDate(now);
}, 200);
