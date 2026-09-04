$(document).ready(function () {

    // ============================================================
    // REFERENCIAS DEL DOM
    // ============================================================

    const ecgFileInput = $('#imageUpload');
    const predictButton = $('#btn-predict');

    const progressSection = $('.progress-section');
    const progressBarFill = $('.progress-bar-fill');
    const progressText = $('.progress-text');

    const resultsDiv = $('#result');

    const imageSection = $('.image-section');
    const mainJumbotron = $('#mainJumbotron');
    const mainContentRow = $('#mainContentRow');

    const patientDataSection = $('#patientDataSection');

    const footerDate = $('#footerDate');
    const devInfo = $('#devInfo');

    const ecgFileToFormatInput = $('#ecgFileToFormat');
    const ecgFileToFormatLabel = $('label[for="ecgFileToFormat"]');

    // En CINC2020 solamente necesitamos el formateo completo.
    const formatEcgFullButton = $('#btn-format-ecg-full');

    let progressInterval = null;


    // ============================================================
    // CONFIGURACIÓN CINC2020
    // ============================================================

    const CINC2020_NUM_CLASSES = 27;
    const CINC2020_NUM_LEADS = 12;
    const CINC2020_SAMPLE_RATE = 500;
    const CINC2020_INPUT_LENGTH = 5000;


    // ============================================================
    // INICIALIZACIÓN
    // ============================================================

    predictButton.hide();
    formatEcgFullButton.hide();

    patientDataSection.hide();

    progressSection.hide();

    ecgFileToFormatLabel.text(
        'Seleccionar archivos WFDB (.mat, .hea, .dat)...'
    );


    // ============================================================
    // EASTER EGG 1 - RICKROLL
    // ============================================================

    footerDate.on('click', function (e) {

        e.preventDefault();

        const youtubeId = 'dQw4w9WgXcQ';

        const youtubeEmbedUrl =
            `https://www.youtube.com/embed/${youtubeId}?autoplay=1&rel=0`;

        $('#rickrollVideo').attr('src', youtubeEmbedUrl);

        $('#rickrollModal').modal('show');
    });


    $('#rickrollModal').on('hide.bs.modal', function () {

        $('#rickrollVideo').attr('src', '');
    });


    // ============================================================
    // EASTER EGG 2 - MEME ALEATORIO
    // ============================================================

    const memeUrls = [
        'https://i.imgflip.com/1ur9b0.jpg',
        'https://i.imgflip.com/39pr6y.jpg',
        'https://i.imgflip.com/2pbmz2.jpg',
        'https://i.imgflip.com/2xlw3q.jpg',
        'https://i.imgflip.com/1g8my4.jpg'
    ];


    devInfo.on('click', function (e) {

        e.preventDefault();

        const randomIndex =
            Math.floor(Math.random() * memeUrls.length);

        window.open(memeUrls[randomIndex], '_blank');
    });


    // ============================================================
    // SELECCIÓN DEL CSV
    // ============================================================

    ecgFileInput.on('change', function () {

        const file = this.files[0];

        if (!file) {
            predictButton.hide();
            return;
        }


        // Verificar extensión

        const fileName = file.name;

        if (!fileName.toLowerCase().endsWith('.csv')) {

            showMessage(
                'Por favor, seleccione un archivo con extensión .csv.',
                'danger'
            );

            ecgFileInput.val('');

            predictButton.hide();

            return;
        }


        // Mostrar nombre

        $(this)
            .next('.custom-file-label')
            .html(fileName);


        // Limpiar resultados anteriores

        resultsDiv.empty();


        // Ocultar información anterior del paciente

        patientDataSection.hide();


        // Reiniciar progreso

        resetProgress();


        // Mostrar botón de predicción

        imageSection.show();
        predictButton.show();


        mainJumbotron.show();
        mainContentRow.show();
        resultsDiv.show();
    });


    // ============================================================
    // SELECCIÓN DE ARCHIVOS WFDB
    // ============================================================

    ecgFileToFormatInput.on('change', function () {

        const files = this.files;

        resultsDiv.empty();

        formatEcgFullButton.hide();


        if (!files || files.length === 0) {

            ecgFileToFormatLabel.text(
                'Seleccionar archivos WFDB (.mat, .hea, .dat)...'
            );

            return;
        }


        const fileNames = [];

        let hasHea = false;
        let hasMat = false;
        let hasDat = false;

        const baseNames = new Set();


        for (let i = 0; i < files.length; i++) {

            const file = files[i];

            fileNames.push(file.name);

            const extension =
                file.name
                    .split('.')
                    .pop()
                    .toLowerCase();

            const lastDot =
                file.name.lastIndexOf('.');

            const baseName =
                lastDot > 0
                    ? file.name.substring(0, lastDot)
                    : file.name;

            baseNames.add(baseName);


            if (extension === 'hea') {
                hasHea = true;
            }

            if (extension === 'mat') {
                hasMat = true;
            }

            if (extension === 'dat') {
                hasDat = true;
            }
        }


        // ========================================================
        // VERIFICAR NOMBRE BASE
        // ========================================================

        if (baseNames.size > 1) {

            showMessage(
                'Todos los archivos WFDB deben tener el mismo nombre base. ' +
                'Por ejemplo: record.mat y record.hea.',
                'danger'
            );

            ecgFileToFormatLabel.text(
                'Seleccionar archivos WFDB (.mat, .hea, .dat)...'
            );

            return;
        }


        // ========================================================
        // VERIFICAR ARCHIVOS
        // ========================================================

        if (!hasHea) {

            showMessage(
                'Para convertir un registro WFDB se necesita el archivo .hea ' +
                'junto con su archivo de datos (.mat o .dat).',
                'warning'
            );

            ecgFileToFormatLabel.html(
                fileNames.join(', ')
            );

            return;
        }


        if (!hasMat && !hasDat) {

            showMessage(
                'Falta el archivo de datos del ECG (.mat o .dat).',
                'danger'
            );

            ecgFileToFormatLabel.html(
                fileNames.join(', ')
            );

            return;
        }


        // ========================================================
        // COMBINACIÓN VÁLIDA
        // ========================================================

        ecgFileToFormatLabel.html(
            fileNames.join(', ')
        );

        formatEcgFullButton.show();
    });


    // ============================================================
    // FORMATEAR WFDB → CSV
    // ============================================================

    formatEcgFullButton.on('click', function () {

        const files = ecgFileToFormatInput[0].files;


        if (!files || files.length === 0) {

            showMessage(
                'Seleccione los archivos WFDB que desea convertir.',
                'warning'
            );

            return;
        }


        formatEcgFullButton.hide();

        startProgress('Formateando ECG...');


        const formData = new FormData();


        for (let i = 0; i < files.length; i++) {

            formData.append(
                'ecg_file_to_format',
                files[i]
            );
        }


        $.ajax({

            type: 'POST',

            url: '/format_ecg_full',

            data: formData,

            contentType: false,

            cache: false,

            processData: false,

            xhrFields: {
                responseType: 'blob'
            },


            success: function (blob, status, xhr) {

                stopProgress(true);


                const disposition =
                    xhr.getResponseHeader(
                        'Content-Disposition'
                    );


                let filename =
                    'formatted_ecg.csv';


                if (
                    disposition &&
                    disposition.indexOf('attachment') !== -1
                ) {

                    const filenameRegex =
                        /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/;

                    const matches =
                        filenameRegex.exec(disposition);


                    if (
                        matches !== null &&
                        matches[1]
                    ) {

                        filename =
                            decodeURIComponent(
                                matches[1]
                                    .replace(/['"]/g, '')
                            );
                    }
                }


                // Crear descarga

                const downloadUrl =
                    window.URL.createObjectURL(blob);

                const a =
                    document.createElement('a');

                a.href = downloadUrl;

                a.download = filename;

                document.body.appendChild(a);

                a.click();

                document.body.removeChild(a);

                window.URL.revokeObjectURL(downloadUrl);


                showMessage(
                    `Archivo "${filename}" convertido correctamente.`,
                    'success'
                );


                // Limpiar input

                ecgFileToFormatInput.val('');

                ecgFileToFormatLabel.text(
                    'Seleccionar archivos WFDB (.mat, .hea, .dat)...'
                );

                formatEcgFullButton.hide();
            },


            error: function (xhr, status, error) {

                stopProgress(false);


                const errorMessage =
                    extractAjaxError(
                        xhr,
                        'Error al convertir el registro WFDB.'
                    );


                showMessage(
                    errorMessage,
                    'danger'
                );


                console.error(
                    'Error en /format_ecg_full:',
                    error,
                    xhr.responseText
                );


                formatEcgFullButton.show();
            }

        });

    });


    // ============================================================
    // BOTONES COLAPSABLES
    // ============================================================

    $(document).on(
        'click',
        '.collapsible',
        function () {

            $(this).toggleClass('active');

            const content =
                $(this).next('.content');

            content.toggleClass('show');
        }
    );


    function initCollapsibles() {

        $('.content')
            .removeClass('show');

        $('.collapsible')
            .removeClass('active');
    }


    // ============================================================
    // PREDICCIÓN
    // ============================================================

    predictButton.on('click', function () {

        const file =
            ecgFileInput[0].files[0];


        if (!file) {

            showMessage(
                'Seleccione un archivo CSV antes de realizar la predicción.',
                'warning'
            );

            return;
        }


        const formData =
            new FormData(
                $('#upload-file')[0]
            );


        predictButton.hide();

        resultsDiv.empty();

        patientDataSection.hide();

        startProgress('Analizando ECG...');


        $.ajax({

            type: 'POST',

            url: '/predict',

            data: formData,

            contentType: false,

            cache: false,

            processData: false,

            async: true,


            success: function (data) {

                console.log(
                    'Respuesta del servidor:',
                    data
                );


                stopProgress(true);


                setTimeout(function () {

                    progressSection.hide();

                    predictButton.show();

                }, 500);


                renderPredictionResults(data);
            },


            error: function (xhr, status, error) {

                stopProgress(false);

                patientDataSection.hide();

                predictButton.show();


                const errorMessage =
                    extractAjaxError(
                        xhr,
                        'Error durante el análisis del ECG.'
                    );


                showMessage(
                    errorMessage,
                    'danger'
                );


                console.error(
                    'Error en /predict:',
                    error,
                    xhr.responseText
                );
            }

        });

    });


    // ============================================================
    // RENDERIZAR RESULTADOS CINC2020
    // ============================================================

    function renderPredictionResults(data) {

        resultsDiv.empty();


        // ========================================================
        // INFORMACIÓN DEL PACIENTE
        // ========================================================

        $('#patientNameResult').text(
            `Nombre del paciente: ${
                data.patient_name || 'No disponible'
            }`
        );


        $('#patientAgeResult').text(
            `Edad del paciente: ${
                data.patient_age || 'No disponible'
            }`
        );


        $('#patientSexResult').text(
            `Sexo del paciente: ${
                data.patient_sex || 'No disponible'
            }`
        );


        patientDataSection.show();


        // ========================================================
        // INFORMACIÓN BÁSICA DEL MODELO
        // ========================================================

        const numClasses =
            data.num_classes || CINC2020_NUM_CLASSES;

        const numLeads =
            data.num_leads || CINC2020_NUM_LEADS;

        const sampleRate =
            data.sampling_rate || CINC2020_SAMPLE_RATE;

        const inputLength =
            data.input_length || CINC2020_INPUT_LENGTH;


        // ========================================================
        // HTML PRINCIPAL
        // ========================================================

        let resultsHtml = `

            <!-- ============================================== -->
            <!-- RESUMEN -->
            <!-- ============================================== -->

            <button
                class="collapsible btn btn-info btn-block mt-4"
                id="generalSummaryCollapsible">

                Resumen de la Predicción

            </button>


            <div
                class="content"
                id="summaryContent">

                <div class="card mt-4">

                    <div class="card-body">

                        <p id="originalLabel"></p>

                        <p id="mostProbableClass"></p>

                        <p id="secondProbableClass"></p>

                        <p id="thirdProbableClass"></p>

                        <p
                            id="cardiacConditionSuggestion">
                        </p>

                        <hr>

                        <p class="small text-muted mb-0">

                            Entrada analizada:
                            ${numLeads} derivaciones,
                            ${sampleRate} Hz,
                            ${inputLength} muestras
                            (${(inputLength / sampleRate).toFixed(1)} s).

                        </p>

                        <p class="small text-muted mb-0">

                            Número de clases evaluadas:
                            ${numClasses}.

                        </p>

                    </div>

                </div>

            </div>


            <!-- ============================================== -->
            <!-- PREDICCIONES -->
            <!-- ============================================== -->

            <button
                class="collapsible btn btn-info btn-block mt-4">

                Predicciones Diagnósticas

            </button>


            <div
                class="content"
                id="diagnosticPredictionsContent">

                <div
                    class="table-responsive"
                    style="max-height: 500px; overflow-y: auto;">

                    <table
                        class="table table-striped table-sm"
                        id="diagnosticPredictionsTable">

                        <thead class="thead-light">

                            <tr>

                                <th>#</th>

                                <th>Código</th>

                                <th>Diagnóstico</th>

                                <th>Probabilidad</th>

                                <th>Predicción</th>

                            </tr>

                        </thead>

                        <tbody>

                        </tbody>

                    </table>

                </div>

            </div>


            <!-- ============================================== -->
            <!-- CLASES DETECTADAS -->
            <!-- ============================================== -->

            <button
                class="collapsible btn btn-info btn-block mt-4">

                Clases Detectadas

            </button>


            <div
                class="content"
                id="positivePredictionsContent">

                <div
                    id="positivePredictions"
                    class="card mt-4">

                    <div class="card-body">

                    </div>

                </div>

            </div>


            <!-- ============================================== -->
            <!-- INFORMACIÓN DE PROBABILIDADES -->
            <!-- ============================================== -->

            <button
                class="collapsible btn btn-info btn-block mt-4">

                Información de Probabilidades

            </button>


            <div
                class="content"
                id="probabilityInfoContent">

                <div class="card mt-4">

                    <div class="card-body">

                        <p>

                            El modelo genera una probabilidad
                            independiente para cada una de las
                            ${numClasses} clases diagnósticas.

                        </p>

                        <p>

                            Una clase se considera predicha cuando
                            su probabilidad supera el umbral utilizado
                            por el modelo.

                        </p>

                        <p class="small text-muted mb-0">

                            Las probabilidades representan la salida
                            del modelo y no constituyen por sí mismas
                            un diagnóstico médico.

                        </p>

                    </div>

                </div>

            </div>


            <!-- ============================================== -->
            <!-- ECG COMPLETO -->
            <!-- ============================================== -->

            <button
                class="collapsible btn btn-info btn-block mt-4">

                ECG de 12 Derivaciones

            </button>


            <div
                class="content"
                id="fullEcgContent">

                <div
                    id="fullEcgPlotContainer"
                    class="card mt-4">

                    <div class="card-body text-center">

                        <div id="fullEcgPlot">

                        </div>

                    </div>

                </div>

            </div>


            <!-- ============================================== -->
            <!-- DESCARGA -->
            <!-- ============================================== -->

            <button
                id="btn-download-ecg-zip"
                class="btn btn-success btn-lg btn-block mt-4">

                Descargar Resultados ECG (ZIP)

            </button>

        `;


        resultsDiv.append(resultsHtml);


        initCollapsibles();


        // ========================================================
        // TABLA DE LAS 27 CLASES
        // ========================================================

        renderDiagnosticPredictions(data);


        // ========================================================
        // CLASES POSITIVAS
        // ========================================================

        renderPositivePredictions(data);


        // ========================================================
        // RESUMEN
        // ========================================================

        renderSummary(data);


        // ========================================================
        // GRÁFICO ECG
        // ========================================================

        renderFullEcgPlot(data);


        // ========================================================
        // BOTÓN DE DESCARGA
        // ========================================================

        configureDownloadButton(data);
    }


    // ============================================================
    // TABLA DE PREDICCIONES
    // ============================================================

    function renderDiagnosticPredictions(data) {

        const tableBody =
            $('#diagnosticPredictionsTable tbody');


        tableBody.empty();


        let predictions =
            Array.isArray(data.predictions)
                ? data.predictions
                : [];


        // Si el backend devuelve average_probabilities
        // como fallback.

        if (
            predictions.length === 0 &&
            Array.isArray(data.average_probabilities)
        ) {

            predictions =
                data.average_probabilities.map(
                    function (probability, index) {

                        return {

                            index: index,

                            code: `Clase ${index + 1}`,

                            class: `Clase ${index + 1}`,

                            description: `Clase ${index + 1}`,

                            probability:
                                probability,

                            predicted:
                                probability >= 0.5
                        };
                    }
                );
        }


        // Ordenar por probabilidad descendente

        predictions.sort(
            function (a, b) {

                return (
                    getProbability(b) -
                    getProbability(a)
                );
            }
        );


        predictions.forEach(
            function (pred, index) {

                const probability =
                    getProbability(pred);


                const percentage =
                    probability * 100;


                const predicted =
                    getPredictedValue(
                        pred,
                        probability
                    );


                const code =
                    pred.code ||
                    pred.snomed_code ||
                    '—';


                const description =
                    pred.description ||
                    pred.class ||
                    pred.name ||
                    'Clase no especificada';


                const row =
                    $('<tr>');


                row.append(
                    $('<td>').text(index + 1)
                );


                row.append(
                    $('<td>').text(code)
                );


                row.append(
                    $('<td>').text(description)
                );


                row.append(
                    $('<td>').text(
                        `${percentage.toFixed(2)}%`
                    )
                );


                const predictionCell =
                    $('<td>');


                if (predicted) {

                    predictionCell
                        .text('Detectada')
                        .addClass(
                            'font-weight-bold'
                        );

                } else {

                    predictionCell.text(
                        'No detectada'
                    );
                }


                row.append(predictionCell);


                tableBody.append(row);
            }
        );
    }


    // ============================================================
    // CLASES POSITIVAS
    // ============================================================

    function renderPositivePredictions(data) {

        const container =
            $('#positivePredictions .card-body');


        container.empty();


        let predictions =
            Array.isArray(data.predictions)
                ? data.predictions
                : [];


        const positivePredictions =
            predictions.filter(
                function (pred) {

                    const probability =
                        getProbability(pred);

                    return getPredictedValue(
                        pred,
                        probability
                    );
                }
            );


        if (positivePredictions.length === 0) {

            container.append(
                $('<p>')
                    .text(
                        'No se superó el umbral de predicción para ninguna clase.'
                    )
                    .addClass('mb-0')
            );

            return;
        }


        const list =
            $('<ul>')
                .addClass('mb-0');


        positivePredictions
            .sort(
                function (a, b) {

                    return (
                        getProbability(b) -
                        getProbability(a)
                    );
                }
            )
            .forEach(
                function (pred) {

                    const probability =
                        getProbability(pred);


                    const code =
                        pred.code ||
                        pred.snomed_code ||
                        '—';


                    const description =
                        pred.description ||
                        pred.class ||
                        pred.name ||
                        'Clase no especificada';


                    const item =
                        $('<li>');


                    item.text(
                        `${code} - ${description}: ` +
                        `${(probability * 100).toFixed(2)}%`
                    );


                    list.append(item);
                }
            );


        container.append(list);
    }


    // ============================================================
    // RESUMEN
    // ============================================================

    function renderSummary(data) {

        const originalLabel =
            $('#originalLabel');

        const mostProbableClass =
            $('#mostProbableClass');

        const secondProbableClass =
            $('#secondProbableClass');

        const thirdProbableClass =
            $('#thirdProbableClass');

        const cardiacSuggestion =
            $('#cardiacConditionSuggestion');


        originalLabel.text(
            `Etiqueta original del registro: ${
                data.original_label ||
                'Desconocida'
            }.`
        );


        if (data.most_probable_class) {

            mostProbableClass.text(
                `Clase con mayor probabilidad: ${
                    data.most_probable_class
                } (${
                    formatCertainty(
                        data.most_probable_certainty
                    )
                }).`
            );

        } else {

            mostProbableClass.text(
                'Clase con mayor probabilidad: N/A.'
            );
        }


        if (data.second_probable_class) {

            secondProbableClass.text(
                `Segunda clase con mayor probabilidad: ${
                    data.second_probable_class
                } (${
                    formatCertainty(
                        data.second_probable_certainty
                    )
                }).`
            );

        } else {

            secondProbableClass.text(
                'Segunda clase con mayor probabilidad: N/A.'
            );
        }


        if (data.third_probable_class) {

            thirdProbableClass.text(
                `Tercera clase con mayor probabilidad: ${
                    data.third_probable_class
                } (${
                    formatCertainty(
                        data.third_probable_certainty
                    )
                }).`
            );

        } else {

            thirdProbableClass.text(
                'Tercera clase con mayor probabilidad: N/A.'
            );
        }


        cardiacSuggestion.text(
            data.cardiac_condition_suggestion ||
            'Las predicciones se muestran como resultados del modelo y no constituyen un diagnóstico clínico.'
        );
    }


    // ============================================================
    // GRÁFICO ECG COMPLETO
    // ============================================================

    function renderFullEcgPlot(data) {

        const container =
            $('#fullEcgPlot');


        container.empty();


        if (!data.full_ecg_plot_url) {

            container.append(
                $('<p>')
                    .addClass('text-muted')
                    .text(
                        'No se generó un gráfico del ECG.'
                    )
            );

            return;
        }


        const image =
            $('<img>')
                .attr(
                    'src',
                    data.full_ecg_plot_url
                )
                .attr(
                    'alt',
                    'ECG de 12 derivaciones'
                )
                .addClass(
                    'img-fluid rounded'
                );


        container.append(image);
    }


    // ============================================================
    // BOTÓN DE DESCARGA
    // ============================================================

    function configureDownloadButton(data) {

        const button =
            $('#btn-download-ecg-zip');


        button.hide();


        if (!data.full_ecg_plot_url) {
            return;
        }


        const urlPath =
            data.full_ecg_plot_url.startsWith('/')
                ? data.full_ecg_plot_url.substring(1)
                : data.full_ecg_plot_url;


        const urlParts =
            urlPath.split('/');


        const recordNameBase =
            urlParts.length > 1
                ? urlParts[1]
                : null;


        if (!recordNameBase) {
            return;
        }


        button
            .data(
                'record-name-base',
                recordNameBase
            )
            .show();


        button.off('click').on(
            'click',
            function () {

                const record =
                    $(this).data(
                        'record-name-base'
                    );


                if (!record) {

                    showMessage(
                        'No se pudo obtener el nombre del registro.',
                        'warning'
                    );

                    return;
                }


                const confirmation =
                    $('<div>')
                        .addClass(
                            'alert alert-info'
                        )
                        .html(
                            '¿Desea descargar los resultados del ECG? ' +
                            '<button class="btn btn-sm btn-primary ml-3" id="confirmDownloadBtn">Sí</button> ' +
                            '<button class="btn btn-sm btn-secondary ml-1" id="cancelDownloadBtn">No</button>'
                        );


                resultsDiv.prepend(
                    confirmation
                );


                $('#confirmDownloadBtn')
                    .on(
                        'click',
                        function () {

                            confirmation.remove();


                            window.location.href =
                                `/download_ecg_results/${encodeURIComponent(record)}`;
                        }
                    );


                $('#cancelDownloadBtn')
                    .on(
                        'click',
                        function () {

                            confirmation.remove();
                        }
                    );
            }
        );
    }


    // ============================================================
    // UTILIDADES
    // ============================================================

    function getProbability(pred) {

        let probability =
            pred.probability;


        if (
            probability === undefined &&
            pred.prob !== undefined
        ) {

            probability =
                pred.prob;
        }


        if (
            probability === undefined &&
            pred.score !== undefined
        ) {

            probability =
                pred.score;
        }


        probability =
            Number(probability);


        if (isNaN(probability)) {
            return 0;
        }


        // Si por alguna razón el backend devuelve
        // porcentaje en lugar de [0,1].

        if (probability > 1) {
            probability /= 100;
        }


        return Math.max(
            0,
            Math.min(1, probability)
        );
    }


    function getPredictedValue(
        pred,
        probability
    ) {

        if (
            pred.predicted !== undefined
        ) {

            return Boolean(
                pred.predicted
            );
        }


        if (
            pred.is_predicted !== undefined
        ) {

            return Boolean(
                pred.is_predicted
            );
        }


        return probability >= 0.5;
    }


    function formatCertainty(value) {

        if (
            value === undefined ||
            value === null
        ) {

            return 'N/A';
        }


        const number =
            Number(value);


        if (isNaN(number)) {
            return 'N/A';
        }


        // El backend puede devolver
        // 0-1 o 0-100.

        const percentage =
            number <= 1
                ? number * 100
                : number;


        return `${percentage.toFixed(1)}%`;
    }


    // ============================================================
    // PROGRESO
    // ============================================================

    function startProgress(message) {

        clearInterval(progressInterval);


        progressSection.show();


        progressBarFill
            .css('width', '0%');


        progressText.text(
            message || 'Procesando...'
        );


        let currentProgress = 0;


        progressInterval =
            setInterval(
                function () {

                    if (currentProgress < 95) {

                        currentProgress += 5;

                        progressBarFill
                            .css(
                                'width',
                                `${currentProgress}%`
                            );


                        progressText.text(
                            `${message || 'Procesando...'} ${currentProgress}%`
                        );
                    }

                },
                300
            );
    }


    function stopProgress(success) {

        clearInterval(
            progressInterval
        );


        if (success) {

            progressBarFill
                .css('width', '100%');


            progressText.text(
                'Completado'
            );

        } else {

            progressBarFill
                .css('width', '0%');


            progressText.text(
                'Error'
            );
        }
    }


    function resetProgress() {

        clearInterval(
            progressInterval
        );


        progressSection.hide();


        progressBarFill
            .css('width', '0%');


        progressText.text(
            '0%'
        );
    }


    // ============================================================
    // MENSAJES
    // ============================================================

    function showMessage(
        message,
        type
    ) {

        const msgBox =
            $('<div>')
                .addClass(
                    `alert alert-${type || 'info'}`
                )
                .text(message);


        resultsDiv.prepend(
            msgBox
        );


        setTimeout(
            function () {

                msgBox.fadeOut(
                    500,
                    function () {
                        $(this).remove();
                    }
                );

            },
            5000
        );
    }


    // ============================================================
    // ERRORES AJAX
    // ============================================================

    function extractAjaxError(
        xhr,
        defaultMessage
    ) {

        if (
            xhr.responseJSON &&
            xhr.responseJSON.error
        ) {

            return xhr.responseJSON.error;
        }


        if (xhr.responseText) {

            try {

                const errorData =
                    JSON.parse(
                        xhr.responseText
                    );


                if (errorData.error) {

                    return errorData.error;
                }

            } catch (e) {

                // No era JSON.
            }
        }


        return defaultMessage;
    }

});