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
    const devInfo = $('#devInfo'); // Referencia al elemento "Desarrollado por"

    let progressInterval;

    // --- Elementos para la nueva sección de formateo de ECG ---
    const ecgFileToFormatInput = $('#ecgFileToFormat');
    const ecgFileToFormatLabel = $('label[for="ecgFileToFormat"]');
    const formatEcgButton = $('#btn-format-ecg');

    // Ocultar el botón de formateo al inicio
    formatEcgButton.hide();
    ecgFileToFormatLabel.text('Seleccionar archivos ECG para formatear (.mat, .hea)...'); // Resetear el texto si es necesario


    // --- EASTER EGG 1: Funcionalidad de Rickroll al hacer clic en la fecha del footer ---
    footerDate.on('click', function(e) {
        e.preventDefault(); 
        const youtubeId = 'dQw4w9WgXcQ'; // ID del video de Rick Astley
        const youtubeEmbedUrl = `https://www.youtube.com/embed/${youtubeId}?autoplay=1&rel=0`; // autoplay=1 para que inicie al abrir, rel=0 para videos relacionados del mismo canal
        
        $('#rickrollVideo').attr('src', youtubeEmbedUrl); // Asigna la URL al iframe
        $('#rickrollModal').modal('show'); // Abre el modal
    });

    // Detener el video cuando el modal de rickroll se cierra
    $('#rickrollModal').on('hide.bs.modal', function () {
        $('#rickrollVideo').attr('src', ''); // Elimina la URL para detener el video
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


    // --- Funcionalidad del input de archivo CSV (para predicción) ---
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

    // --- Funcionalidad del input de archivo ECG (para formatear) ---
    ecgFileToFormatInput.on('change', function(){
        const files = this.files;
        if (files.length > 0) {
            let fileNames = Array.from(files).map(file => file.name).join(', ');
            if (files.length > 2) { // Para evitar un texto demasiado largo
                fileNames = `${files.length} archivos seleccionados`;
            } else if (files.length === 2) {
                fileNames = `${files[0].name}, ${files[1].name}`;
            } else {
                fileNames = files[0].name;
            }
            ecgFileToFormatLabel.html(fileNames); // Actualiza la etiqueta
            formatEcgButton.show(); // Muestra el botón de formatear
        } else {
            ecgFileToFormatLabel.html('Seleccionar archivos ECG para formatear (.mat, .hea)...'); // Restaura la etiqueta
            formatEcgButton.hide(); // Oculta el botón si no hay archivo seleccionado
        }
    });

    // --- Lógica para el botón de Formatear ECG a CSV ---
    formatEcgButton.on('click', function() {
        const files = ecgFileToFormatInput[0].files;
        if (files.length === 0) {
            const msgBox = $('<div>').addClass('alert alert-warning').text('Por favor, selecciona al menos un archivo ECG para formatear.');
            resultsDiv.html(''); 
            resultsDiv.prepend(msgBox);
            setTimeout(() => msgBox.fadeOut(), 3000);
            return;
        }

        // Mostrar indicador de progreso para el formateo
        formatEcgButton.hide();
        progressSection.show(); // Reutilizamos la misma sección de progreso
        progressBarFill.css('width', '0%');
        progressText.text('0%');
        
        let currentFormatProgress = 0;
        progressInterval = setInterval(function() {
            if (currentFormatProgress < 95) {
                currentFormatProgress += 5; 
                progressBarFill.css('width', currentFormatProgress + '%');
                progressText.text(currentFormatProgress + '%');
            }
        }, 150); // Un poco más rápido para el formateo

        const formData = new FormData();
        // Recorre todos los archivos seleccionados y añádelos al FormData
        for (let i = 0; i < files.length; i++) {
            formData.append('ecg_file_to_format', files[i]);
        }

        $.ajax({
            type: 'POST',
            url: '/format_ecg',
            data: formData,
            contentType: false,
            cache: false,
            processData: false,
            xhrFields: {
                responseType: 'blob' // Importante para recibir el archivo como un blob
            },
            success: function (blob, status, xhr) {
                clearInterval(progressInterval);
                progressBarFill.css('width', '100%');
                progressText.text('¡Formateo Completado!');

                // Extraer el nombre del archivo del encabezado Content-Disposition
                const disposition = xhr.getResponseHeader('Content-Disposition');
                let filename = 'formatted_ecg.csv'; // Nombre por defecto
                if (disposition && disposition.indexOf('attachment') !== -1) {
                    const filenameRegex = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/;
                    const matches = filenameRegex.exec(disposition);
                    if (matches != null && matches[1]) {
                        filename = decodeURIComponent(matches[1].replace(/['"]/g, ''));
                    }
                }

                // Crear un enlace temporal y simular un clic para descargar el archivo
                const a = document.createElement('a');
                a.href = window.URL.createObjectURL(blob);
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(a.href); // Limpiar el objeto URL

                const msgBox = $('<div>').addClass('alert alert-success').text(`Archivo "${filename}" formateado y descargado exitosamente.`);
                resultsDiv.html(''); 
                resultsDiv.prepend(msgBox);
                setTimeout(() => msgBox.fadeOut(), 5000);

                // Limpiar el input de archivo y ocultar el botón después de la descarga
                ecgFileToFormatInput.val(''); 
                ecgFileToFormatLabel.html('Seleccionar archivos ECG para formatear (.mat, .hea)...'); 
                setTimeout(function() {
                    progressSection.hide();
                    formatEcgButton.hide(); // Asegurarse de que el botón de formateo esté oculto
                }, 1000); // Dar un tiempo para que el mensaje de éxito se vea

            },
            error: function(xhr, status, error) {
                clearInterval(progressInterval);
                progressBarFill.css('width', '0%');
                progressText.text('Error en Formateo');
                
                setTimeout(function() {
                    progressSection.hide();
                    formatEcgButton.show(); // Mostrar el botón de formateo de nuevo en caso de error
                }, 1000);

                let errorMsg = 'Error al formatear: ' + (xhr.responseJSON ? xhr.responseJSON.error : 'Error desconocido.');
                if (xhr.responseJSON === undefined && xhr.responseText) {
                    try {
                        const errorData = JSON.parse(xhr.responseText);
                        errorMsg = 'Error al formatear: ' + (errorData.error || 'Error desconocido.');
                    } catch (e) {
                        errorMsg = 'Error al formatear: Problema de conexión o archivo inválido.';
                    }
                }

                const msgBox = $('<div>').addClass('alert alert-danger').text(errorMsg);
                resultsDiv.html('');
                resultsDiv.prepend(msgBox);
                setTimeout(() => msgBox.fadeOut(), 5000);
                
                console.error('Error en formateo:', error, xhr.responseText);

                // Limpiar el input de archivo y ocultar el botón en caso de error también
                ecgFileToFormatInput.val('');
                ecgFileToFormatLabel.html('Seleccionar archivos ECG para formatear (.mat, .hea)...');
                formatEcgButton.hide(); // Ocultar el botón después del error para empezar de nuevo
            }
        });
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

                    <!-- NUEVO: Botón para descargar el ZIP de resultados ECG -->
                    <button id="btn-download-ecg-zip" class="btn btn-success btn-lg btn-block mt-4">
                        Descargar Resultados ECG (ZIP)
                    </button>
                `;
                resultsDiv.append(resultsHtml);
                resultsDiv.fadeIn(600);

                // Obtener referencia al nuevo botón de descarga
                const downloadEcgZipButton = $('#btn-download-ecg-zip');

                // Lógica para el botón de descarga
                if (data.full_ecg_plot_url) { // La lógica de descarga depende de que exista al menos un gráfico
                    // Extraer el nombre base del registro de la URL del ECG completo
                    // Se asume que la URL es algo como /resultados/NombreDelRegistro/ECG_Completo_...
                    const urlPath = data.full_ecg_plot_url.startsWith('/') ? data.full_ecg_plot_url.substring(1) : data.full_ecg_plot_url;
                    const urlParts = urlPath.split('/');
                    const recordNameBase = urlParts.length > 1 ? urlParts[1] : null;

                    if (recordNameBase) {
                        // Almacenar el nombre base del registro como un atributo de datos en el botón
                        downloadEcgZipButton.data('record-name-base', recordNameBase);
                        downloadEcgZipButton.show();
                    } else {
                        downloadEcgZipButton.hide();
                    }
                } else {
                    // Si no hay URL del ECG completo, ocultar el botón de descarga
                    downloadEcgZipButton.hide();
                }

                // Event listener para el botón de descarga con confirmación
                downloadEcgZipButton.on('click', function() {
                    // Usar un mensaje box en lugar de alert
                    const msgBoxConfirm = $('<div>').addClass('alert alert-info').html('¿Deseas descargar los gráficos de resultados del ECG en un archivo ZIP? <button class="btn btn-sm btn-primary ml-3" id="confirmDownloadBtn">Sí</button> <button class="btn btn-sm btn-secondary ml-1" id="cancelDownloadBtn">No</button>');
                    resultsDiv.prepend(msgBoxConfirm);

                    $('#confirmDownloadBtn').on('click', function() {
                        msgBoxConfirm.remove();
                        const recordNameBase = downloadEcgZipButton.data('record-name-base');
                        if (recordNameBase) {
                            // Redirigir para iniciar la descarga del ZIP
                            window.location.href = `/download_ecg_results/${recordNameBase}`;
                        } else {
                            const msgBox = $('<div>').addClass('alert alert-warning').text('No se pudo obtener el nombre del registro para la descarga.');
                            resultsDiv.prepend(msgBox);
                            setTimeout(() => msgBox.fadeOut(), 3000);
                        }
                    });

                    $('#cancelDownloadBtn').on('click', function() {
                        msgBoxConfirm.remove();
                    });
                });

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
                        return val === 0 ? '0.0000000%' : val.toFixed(7) + '%';
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
                        row.append(`<td>${(avg_prob * 100).toFixed(7)}%</td>`);
                    });
                } else {
                    console.warn('Advertencia: No se encontraron probabilidades promedio en la respuesta.');
                }

                currentMostProbableClassP.text(`La etiqueta más probable es ${data.most_probable_class || 'N/A'} con una certeza del ${data.most_probable_certainty ? data.most_probable_certainty.toFixed(1) : 'N/A'}%.`);
                currentSecondProbableClassP.text(`La segunda etiqueta prevista es ${data.second_probable_class || 'N/A'} con una certeza del ${data.second_probable_certainty ? data.second_probable_certainty.toFixed(1) : 'N/A'}%.`);
                currentThirdProbableClassP.text(`La tercera etiqueta prevista es ${data.third_probable_class || 'N/A'} con una certeza del ${data.third_probable_certainty ? data.third_probable_certainty.toFixed(1) : 'N/A'}%.`);
                currentOriginalLabelP.text(`La etiqueta original del registro es ${data.original_label || 'N/A'}`);

            },
           error: function(xhr, status, error) {
                clearInterval(progressInterval);
                progressBarFill.css('width', '0%');
                progressText.text('Error');
                
                setTimeout(function() {
                    progressSection.hide();
                    predictButton.show();
                }, 1000);

                // Usar un mensaje box en lugar de alert
                const errorMsg = 'Error en el análisis: ' + (xhr.responseJSON ? xhr.responseJSON.error : 'Error desconocido.');
                const msgBox = $('<div>').addClass('alert alert-danger').text(errorMsg);
                resultsDiv.html(''); // Limpiar antes de añadir el mensaje
                resultsDiv.prepend(msgBox);
                setTimeout(() => msgBox.fadeOut(), 5000); // Hacer que desaparezca después de 5 segundos
                
                console.error('Error:', error);
            }
        });
    });
});
