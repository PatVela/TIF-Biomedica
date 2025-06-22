$(document).ready(function () {
    // Inicialización de elementos del DOM para un acceso más fácil
    const ecgFileInput = $('#imageUpload'); // Input para subir archivos CSV para predicción
    const predictButton = $('#btn-predict'); // Botón de "¡Predecir Ahora!"
    const progressSection = $('.progress-section'); // Sección de la barra de progreso
    const progressBarFill = $('.progress-bar-fill'); // Barra de progreso interior
    const progressText = $('.progress-text'); // Texto que muestra el porcentaje de progreso
    const resultsDiv = $('#result'); // Div donde se mostrarán los resultados de la predicción

    const imageSection = $('.image-section'); // Sección que contiene el botón de predicción
    const mainJumbotron = $('#mainJumbotron'); // Jumbotron principal
    const mainContentRow = $('#mainContentRow'); // Fila principal de contenido

    const fullEcgPlotContainer = $('#fullEcgPlotContainer'); // Contenedor para el gráfico ECG completo (no usado directamente en este JS para gráficos, pero mantenido)
    const footerDate = $('#footerDate'); // Elemento de fecha en el pie de página (para Easter Egg)
    const devInfo = $('#devInfo'); // Referencia al elemento "Desarrollado por" (para Easter Egg)

    let progressInterval; // Variable para almacenar el ID del intervalo de la barra de progreso

    // --- Elementos para la nueva sección de formateo de ECG (WFDB a CSV) ---
    const ecgFileToFormatInput = $('#ecgFileToFormat'); // Input para subir archivos WFDB (.mat, .hea, .dat)
    const ecgFileToFormatLabel = $('label[for="ecgFileToFormat"]'); // Etiqueta del input de archivos WFDB
    const formatEcgButton = $('#btn-format-ecg'); // Botón de "Formatear ECG a CSV"

    // Ocultar el botón de formateo al inicio y establecer el texto por defecto de la etiqueta
    formatEcgButton.hide();
    ecgFileToFormatLabel.text('Seleccionar archivos ECG para formatear (.mat, .hea, .dat)...');


    // --- EASTER EGG 1: Funcionalidad de Rickroll al hacer clic en la fecha del footer ---
    footerDate.on('click', function(e) {
        e.preventDefault(); // Previene la acción por defecto del enlace
        const youtubeId = 'dQw4w9WgXcQ'; // ID del video de Rick Astley
        // URL de incrustación de YouTube con autoplay y sin videos relacionados al final
        const youtubeEmbedUrl = `https://www.youtube.com/embed/${youtubeId}?autoplay=1&rel=0`;

        $('#rickrollVideo').attr('src', youtubeEmbedUrl); // Asigna la URL al iframe del modal
        $('#rickrollModal').modal('show'); // Abre el modal de Rickroll
    });

    // Detener el video cuando el modal de rickroll se cierra
    $('#rickrollModal').on('hide.bs.modal', function () {
        $('#rickrollVideo').attr('src', ''); // Elimina la URL del iframe para detener la reproducción
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
        e.preventDefault(); // Previene la acción por defecto del enlace
        const randomIndex = Math.floor(Math.random() * memeUrls.length); // Selecciona un índice aleatorio
        window.open(memeUrls[randomIndex], '_blank'); // Abre el meme en una nueva pestaña
    });


    // --- Funcionalidad del input de archivo CSV (para predicción) ---
    ecgFileInput.on('change', function(){
        // Obtiene el nombre del archivo seleccionado y lo muestra en la etiqueta
        var fileName = $(this).val().split('\\').pop();
        $(this).next('.custom-file-label').html(fileName);

        // Muestra la sección del botón de predicción y el botón, y limpia resultados anteriores
        imageSection.show();
        predictButton.show();
        resultsDiv.empty();
        fullEcgPlotContainer.empty(); // Limpia el contenedor del gráfico completo si existiera

        // Oculta la barra de progreso y la reinicia
        progressSection.hide();
        progressBarFill.css('width', '0%').text('');
        progressText.text('0%');

        // Asegura que las secciones principales estén visibles
        resultsDiv.show();
        mainJumbotron.show();
        mainContentRow.show();
    });

    // --- Funcionalidad del input de archivo ECG (para formatear) ---
    ecgFileToFormatInput.on('change', function(){
        const files = this.files; // Obtiene la lista de archivos seleccionados
        if (files.length > 0) {
            let fileNames = Array.from(files).map(file => file.name).join(', ');
            // Ajusta el texto de la etiqueta si hay muchos archivos o solo dos
            if (files.length > 2) {
                fileNames = `${files.length} archivos seleccionados`;
            } else if (files.length === 2) {
                fileNames = `${files[0].name}, ${files[1].name}`;
            } else {
                fileNames = files[0].name;
            }
            ecgFileToFormatLabel.html(fileNames); // Actualiza la etiqueta con los nombres de los archivos
            formatEcgButton.show(); // Muestra el botón de formatear
        } else {
            // Si no hay archivos seleccionados, restaura el texto y oculta el botón
            ecgFileToFormatLabel.html('Seleccionar archivos ECG para formatear (.mat, .hea, .dat)...');
            formatEcgButton.hide();
        }
    });

    // --- Lógica para el botón de Formatear ECG a CSV ---
    formatEcgButton.on('click', function() {
        const files = ecgFileToFormatInput[0].files; // Obtiene los archivos del input
        if (files.length === 0) {
            // Muestra un mensaje de advertencia si no se seleccionaron archivos
            const msgBox = $('<div>').addClass('alert alert-warning').text('Por favor, selecciona al menos un archivo ECG para formatear.');
            resultsDiv.html('');
            resultsDiv.prepend(msgBox);
            setTimeout(() => msgBox.fadeOut(), 3000); // El mensaje desaparece después de 3 segundos
            return;
        }

        // Oculta el botón de formateo y muestra la barra de progreso
        formatEcgButton.hide();
        progressSection.show();
        progressBarFill.css('width', '0%');
        progressText.text('0%');

        // Inicia el intervalo de progreso simulado para el formateo
        let currentFormatProgress = 0;
        progressInterval = setInterval(function() {
            if (currentFormatProgress < 95) {
                currentFormatProgress += 5;
                progressBarFill.css('width', currentFormatProgress + '%');
                progressText.text(currentFormatProgress + '%');
            }
        }, 150); // Simula un progreso más rápido para el formateo

        const formData = new FormData();
        // Añade todos los archivos seleccionados al objeto FormData para enviarlos al servidor
        for (let i = 0; i < files.length; i++) {
            formData.append('ecg_file_to_format', files[i]);
        }

        // Realiza la petición AJAX para formatear los archivos
        $.ajax({
            type: 'POST',
            url: '/format_ecg',
            data: formData,
            contentType: false, // Importante para enviar FormData
            cache: false,
            processData: false, // Importante para enviar FormData
            xhrFields: {
                responseType: 'blob' // Espera la respuesta como un blob (archivo)
            },
            success: function (blob, status, xhr) {
                // Detiene el progreso y finaliza la barra
                clearInterval(progressInterval);
                progressBarFill.css('width', '100%');
                progressText.text('¡Formateo Completado!');

                // Extrae el nombre del archivo del encabezado 'Content-Disposition' de la respuesta
                const disposition = xhr.getResponseHeader('Content-Disposition');
                let filename = 'formatted_ecg.csv'; // Nombre por defecto
                if (disposition && disposition.indexOf('attachment') !== -1) {
                    const filenameRegex = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/;
                    const matches = filenameRegex.exec(disposition);
                    if (matches != null && matches[1]) {
                        filename = decodeURIComponent(matches[1].replace(/['"]/g, ''));
                    }
                }

                // Crea un enlace temporal y simula un clic para descargar el archivo
                const a = document.createElement('a');
                a.href = window.URL.createObjectURL(blob);
                a.download = filename;
                document.body.appendChild(a);
                a.click(); // Simula el clic para iniciar la descarga
                document.body.removeChild(a); // Elimina el enlace temporal
                window.URL.revokeObjectURL(a.href); // Libera la URL del objeto blob

                // Muestra un mensaje de éxito
                const msgBox = $('<div>').addClass('alert alert-success').text(`Archivo "${filename}" formateado y descargado exitosamente.`);
                resultsDiv.html('');
                resultsDiv.prepend(msgBox);
                setTimeout(() => msgBox.fadeOut(), 5000); // El mensaje desaparece después de 5 segundos

                // Limpia el input de archivo y oculta el botón después de la descarga exitosa
                ecgFileToFormatInput.val('');
                ecgFileToFormatLabel.html('Seleccionar archivos ECG para formatear (.mat, .hea, .dat)...');
                setTimeout(function() {
                    progressSection.hide();
                    formatEcgButton.hide(); // Asegurarse de que el botón de formateo esté oculto
                }, 1000);

            },
            error: function(xhr, status, error) {
                // En caso de error, detiene el progreso y muestra el botón de nuevo
                clearInterval(progressInterval);
                progressBarFill.css('width', '0%');
                progressText.text('Error en Formateo');

                setTimeout(function() {
                    progressSection.hide();
                    formatEcgButton.show(); // Muestra el botón de formateo de nuevo
                }, 1000);

                // Intenta parsear el mensaje de error del servidor o usa un mensaje genérico
                let errorMsg = 'Error al formatear: ' + (xhr.responseJSON ? xhr.responseJSON.error : 'Error desconocido.');
                if (xhr.responseJSON === undefined && xhr.responseText) {
                    try {
                        const errorData = JSON.parse(xhr.responseText);
                        errorMsg = 'Error al formatear: ' + (errorData.error || 'Error desconocido.');
                    } catch (e) {
                        errorMsg = 'Error al formatear: Problema de conexión o archivo inválido.';
                    }
                }

                // Muestra un mensaje de error
                const msgBox = $('<div>').addClass('alert alert-danger').text(errorMsg);
                resultsDiv.html('');
                resultsDiv.prepend(msgBox);
                setTimeout(() => msgBox.fadeOut(), 5000);

                console.error('Error en formateo:', error, xhr.responseText);

                // Limpia el input de archivo y oculta el botón en caso de error
                ecgFileToFormatInput.val('');
                ecgFileToFormatLabel.html('Seleccionar archivos ECG para formatear (.mat, .hea, .dat)...');
                formatEcgButton.hide(); // Oculta el botón después del error para empezar de nuevo
            }
        });
    });


    // --- Funcionalidad de los botones colapsables (para mostrar/ocultar resultados) ---
    $(document).on('click', '.collapsible', function() {
        $(this).toggleClass("active"); // Alterna la clase 'active' para el estilo
        var content = $(this).next('.content'); // Obtiene el contenido colapsable siguiente
        if (content.css('max-height') !== '0px' && content.css('max-height') !== '') {
            content.css('max-height', '0'); // Si está abierto, lo cierra
        } else {
            // Si está cerrado, lo abre estableciendo la altura máxima a su altura de desplazamiento
            content.css('max-height', content.prop('scrollHeight') + 30 + 'px');
        }
    });

    // --- Lógica de Predicción (cuando se sube un archivo CSV) ---
    predictButton.on('click', function () {
        var form_data = new FormData($('#upload-file')[0]); // Crea un objeto FormData con el archivo CSV

        $(this).hide(); // Oculta el botón de predicción
        progressSection.show(); // Muestra la barra de progreso

        let currentProgress = 0;
        progressBarFill.css('width', currentProgress + '%'); // Reinicia la barra de progreso
        progressText.text(currentProgress + '%');

        // Inicia el intervalo de progreso simulado para la predicción
        progressInterval = setInterval(function() {
            if (currentProgress < 95) {
                currentProgress += 5;
                progressBarFill.css('width', currentProgress + '%');
                progressText.text(currentProgress + '%');
            }
        }, 300); // Simula un progreso más lento para la predicción

        resultsDiv.empty(); // Limpia los resultados anteriores
        resultsDiv.show(); // Asegura que el área de resultados esté visible
        fullEcgPlotContainer.empty(); // Limpia el contenedor del gráfico completo

        mainJumbotron.show(); // Asegura que el jumbotron esté visible
        mainContentRow.show(); // Asegura que la fila de contenido principal esté visible

        // Realiza la petición AJAX al servidor para la predicción
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true, // Petición asíncrona
            success: function (data) {
                // En caso de éxito, detiene el progreso y finaliza la barra
                clearInterval(progressInterval);
                progressBarFill.css('width', '100%');
                progressText.text('100%');

                setTimeout(function() {
                    progressSection.hide(); // Oculta la barra de progreso
                    predictButton.show(); // Muestra el botón de predicción de nuevo
                }, 500);

                // Genera el HTML para mostrar los resultados de la predicción
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
                                <p id="accuracyResult" style="font-weight: bold;"></p>
                                <p id="f1ScoreResult" style="font-weight: bold;"></p>
                            </div>
                        </div>
                    </div>

                    <!-- Botón para descargar el ZIP de resultados ECG -->
                    <button id="btn-download-ecg-zip" class="btn btn-success btn-lg btn-block mt-4">
                        Descargar Resultados ECG (ZIP)
                    </button>
                `;
                resultsDiv.append(resultsHtml); // Añade el HTML al div de resultados
                resultsDiv.fadeIn(600); // Muestra los resultados con un efecto de fade

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
                        downloadEcgZipButton.show(); // Muestra el botón de descarga
                    } else {
                        downloadEcgZipButton.hide(); // Oculta el botón si no se puede obtener el nombre base
                    }
                } else {
                    // Si no hay URL del ECG completo, ocultar el botón de descarga
                    downloadEcgZipButton.hide();
                }

                // Event listener para el botón de descarga con confirmación
                downloadEcgZipButton.on('click', function() {
                    // Muestra un cuadro de mensaje de confirmación en lugar de un alert nativo
                    const msgBoxConfirm = $('<div>').addClass('alert alert-info').html('¿Deseas descargar los gráficos de resultados del ECG en un archivo ZIP? <button class="btn btn-sm btn-primary ml-3" id="confirmDownloadBtn">Sí</button> <button class="btn btn-sm btn-secondary ml-1" id="cancelDownloadBtn">No</button>');
                    resultsDiv.prepend(msgBoxConfirm); // Añade el cuadro de confirmación al inicio del div de resultados

                    // Manejador del clic para el botón "Sí" de la confirmación
                    $('#confirmDownloadBtn').on('click', function() {
                        msgBoxConfirm.remove(); // Elimina el cuadro de confirmación
                        const recordNameBase = downloadEcgZipButton.data('record-name-base');
                        if (recordNameBase) {
                            // Redirige para iniciar la descarga del ZIP
                            window.location.href = `/download_ecg_results/${recordNameBase}`;
                        } else {
                            // Muestra un mensaje de advertencia si no se pudo obtener el nombre del registro
                            const msgBox = $('<div>').addClass('alert alert-warning').text('No se pudo obtener el nombre del registro para la descarga.');
                            resultsDiv.prepend(msgBox);
                            setTimeout(() => msgBox.fadeOut(), 3000);
                        }
                    });

                    // Manejador del clic para el botón "No" de la confirmación
                    $('#cancelDownloadBtn').on('click', function() {
                        msgBoxConfirm.remove(); // Elimina el cuadro de confirmación
                    });
                });

                // Obtiene las referencias a los elementos de las tablas y párrafos para rellenar los datos
                const currentDetailedPredictionsTableBody = $('#detailedPredictionsTable tbody');
                const currentAveragePredictionTableBody = $('#averagePredictionTable tbody');
                const currentMostProbableClassP = $('#mostProbableClass');
                const currentSecondProbableClassP = $('#secondProbableClass');
                const currentThirdProbableClassP = $('#thirdProbableClass');
                const currentOriginalLabelP = $('#originalLabel');
                const currentAccuracyP = $('#accuracyResult'); // Nuevo
                const currentF1ScoreP = $('#f1ScoreResult');   // Nuevo
                const currentSegmentPlotsContainer = $('#segmentPlotsContainer'); // No se usa aquí para añadir plots, pero se mantiene la referencia

                currentDetailedPredictionsTableBody.empty(); // Limpia la tabla de predicciones detalladas
                currentAveragePredictionTableBody.empty(); // Limpia la tabla de predicciones promedio
                currentSegmentPlotsContainer.empty(); // Limpia el contenedor de gráficos de segmento

                // Rellena la tabla de predicciones detalladas
                data.predictions.forEach(function(pred, index) {
                    const row = $('<tr>').appendTo(currentDetailedPredictionsTableBody);
                    // Formatea las probabilidades para mostrar con precisión completa
                    // Usar 'all_probs_string' que ya viene formateado
                    const probabilitiesFormatted = pred.all_probs_string; 

                    row.append(`<td>${index + 1}</td>`);
                    row.append(`<td>${pred.class}</td>`);
                    row.append(`<td>${probabilitiesFormatted}</td>`);
                });

                // Mapeo de índices a nombres de clase para la tabla de promedio
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

                // Actualiza los párrafos de resumen general
                currentMostProbableClassP.text(`La etiqueta más probable es ${data.most_probable_class || 'N/A'} con una certeza del ${data.most_probable_certainty ? data.most_probable_certainty.toFixed(1) : 'N/A'}%.`);
                currentSecondProbableClassP.text(`La segunda etiqueta prevista es ${data.second_probable_class || 'N/A'} con una certeza del ${data.second_probable_certainty ? data.second_probable_certainty.toFixed(1) : 'N/A'}%.`);
                currentThirdProbableClassP.text(`La tercera etiqueta prevista es ${data.third_probable_class || 'N/A'} con una certeza del ${data.third_probable_certainty ? data.third_probable_certainty.toFixed(1) : 'N/A'}%.`);
                currentOriginalLabelP.text(`La etiqueta original del registro es ${data.original_label || 'N/A'}`);
                // Mostrar Accuracy y F1-score solo si están disponibles
                if (data.accuracy_val) {
                    currentAccuracyP.text(`Accuracy del Modelo: ${data.accuracy_val}`);
                } else {
                    currentAccuracyP.text(''); // Limpiar si no hay datos
                }
                if (data.f1_score_val) {
                    currentF1ScoreP.text(`F1-score del Modelo (ponderado): ${data.f1_score_val}`);
                } else {
                    currentF1ScoreP.text(''); // Limpiar si no hay datos
                }

            },
           error: function(xhr, status, error) {
                // En caso de error, detiene el progreso y muestra el botón de predicción
                clearInterval(progressInterval);
                progressBarFill.css('width', '0%');
                progressText.text('Error');

                setTimeout(function() {
                    progressSection.hide();
                    predictButton.show();
                }, 1000);

                // Muestra un mensaje de error utilizando un cuadro de mensaje personalizado
                const errorMsg = 'Error en el análisis: ' + (xhr.responseJSON ? xhr.responseJSON.error : 'Error desconocido.');
                const msgBox = $('<div>').addClass('alert alert-danger').text(errorMsg);
                resultsDiv.html(''); // Limpia antes de añadir el mensaje
                resultsDiv.prepend(msgBox); // Añade el mensaje al inicio
                setTimeout(() => msgBox.fadeOut(), 5000); // El mensaje desaparece después de 5 segundos

                console.error('Error:', error); // Imprime el error en la consola
            }
        });
    });
});