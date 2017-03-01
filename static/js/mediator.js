
var BASEURL = 'http://localhost:8000/api/1.0/'

function makePIE(positive, negative){
    $("#chartContainer").CanvasJSChart({
		title: {
			text: "Summary",
			fontSize: 24
		},
		axisY: {
			title: "Sentiment in %"
		},
		legend :{
			verticalAlign: "center",
			horizontalAlign: "right"
		},
		data: [
		{
			type: "pie",
			showInLegend: true,
			toolTipContent: "{label} <br/> {y} %",
			indexLabel: "{y} %",
			dataPoints: [
				{ label: "Positive",  y: positive, legendText: "Positive"},
				{ label: "Negative",    y: negative, legendText: "Negative"  }
			]
		}
		]
	});

	$('.canvasjs-chart-canvas').css({"position":"relative"});
/*
	$("canvas:last").remove();

	$('.canvasjs-chart-canvas[style="product_id"]').remove();
*/

	$('.canvasjs-chart-container canvas:last').remove();
}


function makepiechart(positive, negative){
    var data = [
    ['Positive', positive],['Negative', negative]  ];
  var plot2 = jQuery.jqplot ('chart2', [data],
    {
      seriesDefaults: {
        renderer: jQuery.jqplot.PieRenderer,
        rendererOptions: {
          // Turn off filling of slices.
          fill: false,
          showDataLabels: true,
          // Add a margin to seperate the slices.
          sliceMargin: 4,
          // stroke the slices with a little thicker line.
          lineWidth: 5
        }
      },
      legend: { show:true, location: 'e' }
    }
  );
}



function make_sentiment_list(data){
    var positive_result = '';
    data_positive = data["positive"];
    data_negative = data["negative"];

    $("#negationside").show();
    $("#positiveside").show();

    for(var i=0; i < data_positive.length; i++){
        positive_result += '<li style="height:15px:">' + data_positive[i] + '</li>' + '<hr>';

    }
    $("#ulPositive").html(positive_result);


    var negative_result = '';
    for(var i=0; i < data_negative.length; i++){
        negative_result += '<li style="height:15px:">' + data_negative[i] + '</li>' + '<hr>';

    }

    $("#ulNegative").html(negative_result);


    var total_data = data_positive.length + data_negative.length;

    var percentage_positive = (data_positive.length * 100)/total_data;

    var percentage_negative = (data_negative.length * 100)/total_data;

    //makepiechart(percentage_positive, percentage_negative)
    makePIE(percentage_positive, percentage_negative)
}




$("#startAnalysis").click(function(e){

   end_url = BASEURL + 'startanalysisapi';
    $.ajax({
        type: 'GET',
        url: end_url,
        dataType:'json',
        contentType:'application/json',
        success: function(data){
            make_sentiment_list(data);

        },
        error:function(e){
        }
    });
    //window.location.href('http://localhost:8000/api/1.0/startanalysisapi');
});
/*

    $('#startAnalysis').click(function(e){
        alert("start analysis");
        jQuery.ajax({
            type: 'GET',
            url: 'http://localhost:8000/api/1.0/startanalysisapi',
            contentType: "application/json; charset=utf-8",
            dataType: 'json',
            success: function(data){
                alert("success test");
                */
/*if data['status'] == 1:
                    $('#voltdbstatus').text('Cluster Running')*//*

            },
            error: function (e) {
                alert("error test");
                console.log(e.message);
            }

       });
    });


*/

function find_url(){
    var review_url = $("#reviewurl").val();
    return review_url;
}

$("#btnAddReviewUrl").click(function(e){

    var review_url = find_url();
    var request_data = {"reviewurl":review_url};
    $.ajax({
        type: 'POST',
        url:  BASEURL + 'downloadreviewsapi',
        dataType:'json',
        contentType:'application/json',
        data: JSON.stringify({
            'reviewurl':review_url
        }),
        success: function(data){
            make_sentiment_list(data);

        },
        error:function(e){
        }
    });

});



