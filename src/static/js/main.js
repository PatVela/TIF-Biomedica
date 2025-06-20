$(document).ready(function () {
    // Inicialización de elementos
    const ecgFileInput = $('#imageUpload');
    const predictButton = $('#btn-predict');
    const progressSection = $('.progress-section');
    const progressBarFill = $('.progress-bar-fill');
    const progressText = $('.progress-text');
    const resultsDiv = $('#result');

    const imageSection = $('.image-section');
    const mainJumbotron = $('#mainJumbotron');
    const mainContentRow = $('#mainContentRow');

    const fullEcgPlotContainer = $('#fullEcgPlotContainer');
    const footerDate = $('#footerDate'); 
    const devInfo = $('#devInfo'); // Nuevo: Referencia al elemento "Desarrollado por"

    let progressInterval;

    // --- EASTER EGG 1: Funcionalidad de Rickroll al hacer clic en la fecha del footer ---
    footerDate.on('click', function(e) {
        e.preventDefault(); 
        window.open('https://www.youtube.com/watch?v=dQw4w9WgXcQ', '_blank'); 
    });

    // --- EASTER EGG 2: Meme aleatorio al hacer clic en "Desarrollado por JPDV y EMOG." ---
    const memeUrls = [
        'https://i.imgflip.com/1ur9b0.jpg', // Distracted Boyfriend
        'https://i.imgflip.com/39pr6y.jpg', // Woman Yelling at Cat
        'https://i.imgflip.com/2pbmz2.jpg', // Expanding Brain
        'https://i.imgflip.com/2xlw3q.jpg', // Drake Hotline Bling
        'https://i.imgflip.com/1g8my4.jpg'  // Success Kid
    ];

    devInfo.on('click', function(e) {
        e.preventDefault();
        const randomIndex = Math.floor(Math.random() * memeUrls.length);
        window.open(memeUrls[randomIndex], '_blank');
    });


    // --- Funcionalidad del input de archivo ---
    ecgFileInput.on('change', function(){
        var fileName = $(this).val().split('\\').pop();
        $(this).next('.custom-file-label').html(fileName);
        imageSection.show();
        predictButton.show();
        resultsDiv.empty();
        fullEcgPlotContainer.empty();

        progressSection.hide();
        progressBarFill.css('width', '0%').text('');
        progressText.text('0%');

        resultsDiv.show();
        mainJumbotron.show();
        mainContentRow.show();
    });

    // --- Funcionalidad de los botones colapsables ---
    $(document).on('click', '.collapsible', function() {
        $(this).toggleClass("active");
        var content = $(this).next('.content');
        if (content.css('max-height') !== '0px' && content.css('max-height') !== '') {
            content.css('max-height', '0');
        } else {
            content.css('max-height', content.prop('scrollHeight') + 30 + 'px'); 
        } 
    });

    // --- Lógica de Predicción ---
    predictButton.on('click', function () {
        var form_data = new FormData($('#upload-file')[0]);

        $(this).hide();
        progressSection.show();

        let currentProgress = 0;
        progressBarFill.css('width', currentProgress + '%');
        progressText.text(currentProgress + '%');

        progressInterval = setInterval(function() {
            if (currentProgress < 95) {
                currentProgress += 5; 
                progressBarFill.css('width', currentProgress + '%');
                progressText.text(currentProgress + '%');
            }
        }, 300);
        
        resultsDiv.empty(); 
        resultsDiv.show(); 
        fullEcgPlotContainer.empty();

        mainJumbotron.show();
        mainContentRow.show();
        
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                clearInterval(progressInterval);
                progressBarFill.css('width', '100%');
                progressText.text('100%');
                
                setTimeout(function() {
                    progressSection.hide();
                    predictButton.show();
                }, 500);

                let resultsHtml = `
                    <button class="collapsible btn btn-info btn-block mt-4">Análisis por Segmento</button>
                    <div class="content" id="detailedPredictionsContent">
                        <div style="max-height: 400px; overflow-y: auto;">
                            <table class="table table-striped" id="detailedPredictionsTable">
                                <thead class="thead-light">
                                    <tr>
                                        <th>Segmento #</th>
                                        <th>Clase Detectada</th>
                                        <th>Probabilidad</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <!-- Filas a insertar por JS -->
                                </tbody>
                            </table>
                        </div>
                    </div>

                    <button class="collapsible btn btn-info btn-block mt-4">Media de la Predicción</button>
                    <div class="content" id="averagePredictionContent">
                        <table class="table table-striped" id="averagePredictionTable">
                            <thead class="thead-light">
                                <tr>
                                    <th>Clase</th>
                                    <th>Probabilidad Media</th>
                                </tr>
                            </thead>
                            <tbody>
                                <!-- Filas a insertar por JS -->
                            </tbody>
                        </table>
                    </div>

                    <button class="collapsible btn btn-info btn-block mt-4">Resumen General</button>
                    <div class="content" id="summaryContent">
                        <div class="card mt-4">
                            <div class="card-header">
                                <strong>Resumen General</strong>
                            </div>
                            <div class="card-body">
                                <p id="mostProbableClass"></p>
                                <p id="secondProbableClass"></p>
                                <p id="thirdProbableClass"></p>
                                <p id="originalLabel"></p>
                            </div>
                        </div>
                    </div>

                    <!-- Botón para mostrar los ECGs de cada segmento -->
                    <button class="collapsible btn btn-info btn-block mt-4">Mostrar ECGs de Segmentos</button>
                    <div class="content">
                        <div id="segmentPlotsContainer" class="collapsible-inner-content">
                            <!-- Los gráficos de cada segmento se cargarán aquí por JS -->
                        </div>
                    </div>
                `;
                resultsDiv.append(resultsHtml);
                resultsDiv.fadeIn(600);

                const currentDetailedPredictionsTableBody = $('#detailedPredictionsTable tbody');
                const currentAveragePredictionTableBody = $('#averagePredictionTable tbody');
                const currentMostProbableClassP = $('#mostProbableClass');
                const currentSecondProbableClassP = $('#secondProbableClass');
                const currentThirdProbableClassP = $('#thirdProbableClass');
                const currentOriginalLabelP = $('#originalLabel');
                const currentSegmentPlotsContainer = $('#segmentPlotsContainer');

                currentDetailedPredictionsTableBody.empty();
                currentAveragePredictionTableBody.empty();
                currentSegmentPlotsContainer.empty();

                data.predictions.forEach(function(pred, index) {
                    const row = $('<tr>').appendTo(currentDetailedPredictionsTableBody);
                    const probParts = pred.probability.split(' | ').map(p => {
                        const val = parseFloat(p.replace('%', ''));
                        return val === 0 ? '0.0000000e+0%' : val.toExponential(7) + '%';
                    });
                    const probabilitiesFormatted = probParts.join(' | ');
                    
                    row.append(`<td>${index + 1}</td>`);
                    row.append(`<td>${pred.class}</td>`);
                    row.append(`<td>${probabilitiesFormatted}</td>`);
                });

                const classesMap = ['Normal', 'Ventricular', 'Estimulado', 'Auricular', 'Fusión', 'Ruido'];
                if (data.average_probabilities) {
                    data.average_probabilities.forEach(function(avg_prob, index) {
                        const row = $('<tr>').appendTo(currentAveragePredictionTableBody);
                        row.append(`<td>${classesMap[index]}</td>`);
                        row.append(`<td>${avg_prob.toExponential(7)}%</td>`);
                    });
                } else {
                    console.warn('Advertencia: No se encontraron probabilidades promedio en la respuesta.');
                }

                currentMostProbableClassP.text(`La etiqueta más probable es ${data.most_probable_class || 'N/A'} con una certeza del ${data.most_probable_certainty ? data.most_probable_certainty.toFixed(1) : 'N/A'}%.`);
                currentSecondProbableClassP.text(`La segunda etiqueta prevista es ${data.second_probable_class || 'N/A'} con una certeza del ${data.second_probable_certainty ? data.second_probable_certainty.toFixed(1) : 'N/A'}%.`);
                currentThirdProbableClassP.text(`La tercera etiqueta prevista es ${data.third_probable_class || 'N/A'} con una certeza del ${data.third_probable_certainty ? data.third_probable_certainty.toFixed(1) : 'N/A'}%.`);
                currentOriginalLabelP.text(`La etiqueta original del registro es ${data.original_label || 'N/A'}`);

                if (data.full_ecg_plot_url) {
                    const img = $('<img>').attr({
                        src: data.full_ecg_plot_url,
                        alt: 'ECG Completo'
                    }).addClass('segment-image'); 
                    fullEcgPlotContainer.append(img); 
                } else {
                    fullEcgPlotContainer.html('<p class="text-red-400">No se pudo cargar el gráfico del ECG completo.</p>');
                }

                if (data.segment_plot_urls && data.segment_plot_urls.length > 0) {
                    data.segment_plot_urls.forEach(function(url) {
                        const img = $('<img>').attr({
                            src: url,
                            alt: 'Segmento ECG'
                        }).addClass('segment-image');
                        currentSegmentPlotsContainer.append(img);
                    });
                } else {
                    currentSegmentPlotsContainer.html('<p class="text-red-400">No se pudieron cargar los gráficos de segmentos.</p>');
                }
            },
            error: function(xhr, status, error) {
                clearInterval(progressInterval);
                progressBarFill.css('width', '0%');
                progressText.text('Error');
                
                setTimeout(function() {
                    progressSection.hide();
                    predictButton.show();
                }, 1000);

                resultsDiv.html('Error en el análisis: ' + (xhr.responseJSON ? xhr.responseJSON.error : 'Error desconocido.'));
                resultsDiv.fadeIn(600);
                console.error('Error:', error);
            }
        });
    });
});
