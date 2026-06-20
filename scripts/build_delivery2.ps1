$ErrorActionPreference = 'Stop'

$root = Resolve-Path (Join-Path $PSScriptRoot '..')
$src = Join-Path $root 'DELIVERABLE_2/jose_things'
$dataSrc = Join-Path $root 'DELIVERABLE_2/nlp_d2_data'
$dst = Join-Path $root 'DELIVERABLE_2/JoseCalatayud_MiquelAngelLlauger_LlucSegura'

$dirs = @($dst, (Join-Path $dst 'data'), (Join-Path $dst 'fitted_models'),
    (Join-Path $dst 'utils'), (Join-Path $dst 'source_notebooks'))
foreach ($dir in $dirs) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }

Copy-Item (Join-Path $dataSrc '*.csv') (Join-Path $dst 'data') -Force
Copy-Item (Join-Path $src 'fitted_models/*') (Join-Path $dst 'fitted_models') -Recurse -Force
Copy-Item (Join-Path $src 'environment.yml') $dst -Force
Copy-Item (Join-Path $src 'requirements.txt') $dst -Force
Copy-Item (Join-Path $src 'report_outline.md') $dst -Force

$sourceNames = @(
    '01_EDA.ipynb', '02_Model_CRF.ipynb', '03_Model_DL_BiLSTM.ipynb',
    '03_1_BiLSTM_CRF.ipynb', '03_2_BiLSTM_GloVe.ipynb',
    '03_3_BiLSTM_GloVe_CRF.ipynb', '04_Model_DL_Transformer.ipynb'
)
foreach ($name in $sourceNames) {
    Copy-Item (Join-Path $src $name) (Join-Path $dst 'source_notebooks') -Force
}

function Update-NotebookPaths($notebook) {
    foreach ($cell in $notebook.cells) {
        if ($null -eq $cell.source) { continue }
        for ($i = 0; $i -lt $cell.source.Count; $i++) {
            $line = [string]$cell.source[$i]
            $line = $line.Replace("DATA_DIR = '../nlp_d2_data'", "DATA_DIR = 'data'")
            $line = $line.Replace("DATA_DIR = '.'", "DATA_DIR = 'data'")
            $line = $line.Replace("DATA_DIR = 'data' if IN_COLAB else '../nlp_d2_data'", "DATA_DIR = 'data'")
            $line = $line.Replace('05_reproduce_results.ipynb', 'reproduce_results.ipynb')
            $cell.source[$i] = $line
        }
    }
}

# Consolidate all analysis/training work into the exact notebook name required by the guide.
$first = Get-Content -Raw (Join-Path $src $sourceNames[0]) | ConvertFrom-Json
Update-NotebookPaths $first
$cells = New-Object System.Collections.ArrayList
$intro = [pscustomobject]@{
    cell_type = 'markdown'; metadata = [pscustomobject]@{}; source = @(
        "# Train models`n", "`n",
        "This notebook consolidates the complete EDA and training pipeline required for Deliverable 2.`n",
        "Run it from top to bottom in this folder. It writes every artifact to `fitted_models/`.`n",
        "Sections retain the original notebook boundaries for traceability.`n"
    )
}
[void]$cells.Add($intro)
for ($n = 0; $n -lt $sourceNames.Count; $n++) {
    $nb = Get-Content -Raw (Join-Path $src $sourceNames[$n]) | ConvertFrom-Json
    Update-NotebookPaths $nb
    $boundary = [pscustomobject]@{
        cell_type = 'markdown'; metadata = [pscustomobject]@{}; source = @(
            "---`n", "# Source section: $($sourceNames[$n])`n"
        )
    }
    [void]$cells.Add($boundary)
    foreach ($cell in $nb.cells) { [void]$cells.Add($cell) }
}
$first.cells = $cells
$first | ConvertTo-Json -Depth 100 -Compress | Set-Content -Encoding UTF8 (Join-Path $dst 'train_models.ipynb')

$reproduce = Get-Content -Raw (Join-Path $src '05_reproduce_results.ipynb') | ConvertFrom-Json
Update-NotebookPaths $reproduce
$reproduce.cells[0].source = @(
    "# Reproduce results`n", "`n",
    "Loads the supplied data and fitted artifacts without training, then reports train/test non-O accuracy, weighted F1, confusion matrices, and TINY TEST predictions in the required `word/tag` format.`n",
    "`n", "If an optional artifact is absent or incomplete, the notebook identifies it explicitly and continues with the available models.`n"
)
$reproduce | ConvertTo-Json -Depth 100 -Compress | Set-Content -Encoding UTF8 (Join-Path $dst 'reproduce_results.ipynb')

$utils = @'
"""Shared data loading and evaluation helpers for the NER delivery."""
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def load_sentences(path):
    """Load a token-level CSV and return ordered sentences of (word, tag) pairs."""
    frame = pd.read_csv(Path(path))
    required = {"sentence_id", "words", "tags"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame["words"] = frame["words"].astype(str)
    frame["tags"] = frame["tags"].astype(str)
    frame["sentence_id"] = frame["sentence_id"].astype(int)
    return [list(zip(group["words"], group["tags"]))
            for _, group in frame.groupby("sentence_id", sort=True)]


def flatten(sequences):
    return [item for sequence in sequences for item in sequence]


def non_o_accuracy(y_true, y_pred):
    """Token accuracy restricted to gold labels other than O, as required."""
    true_flat, pred_flat = flatten(y_true), flatten(y_pred)
    selected = [i for i, tag in enumerate(true_flat) if tag != "O"]
    return accuracy_score([true_flat[i] for i in selected],
                          [pred_flat[i] for i in selected])


def weighted_non_o_f1(y_true, y_pred):
    """Weighted token F1 over entity labels, excluding O and padding."""
    true_flat, pred_flat = flatten(y_true), flatten(y_pred)
    labels = sorted(set(true_flat).difference({"O", "<PAD>"}))
    return f1_score(true_flat, pred_flat, labels=labels,
                    average="weighted", zero_division=0)


def tagged_sentence(sentence, predictions):
    """Format one prediction in the guide's w1/t1 w2/t2 form."""
    return " ".join(f"{word}/{tag}" for (word, _), tag in zip(sentence, predictions))
'@
Set-Content -Encoding UTF8 (Join-Path $dst 'utils/utils.py') $utils
Set-Content -Encoding ASCII (Join-Path $dst 'utils/__init__.py') ''

$readme = @'
# NER - Deliverable 2

Team: Jose Calatayud, Miquel __AGRAVE__ngel Llauger, Lluc Segura

This package follows the required submission layout. `train_models.ipynb` contains the
complete EDA and training pipeline for three CRFs, BiLSTM, BiLSTM-CRF, BiLSTM-GloVe,
BiLSTM-GloVe-CRF, and DistilBERT. `reproduce_results.ipynb` performs evaluation only and
loads artifacts from `fitted_models/`.

## Reproduction

```bash
conda env create -f environment.yml
conda activate nlp_d2
jupyter nbconvert --to notebook --execute --inplace reproduce_results.ipynb
```

For retraining, execute `train_models.ipynb` from top to bottom. GPU execution is strongly
recommended for the neural models. GloVe training may download pretrained vectors.

The `source_notebooks/` directory preserves the original, individually executable stages.
The included DistilBERT directory contains its configuration and tokenizer; if weight files
are unavailable, the reproduction notebook skips that model and evaluates the remaining
supplied artifacts.
'@
$readme = $readme.Replace([string]'__AGRAVE__', [string][char]0x00C0)
Set-Content -Encoding UTF8 (Join-Path $dst 'README.md') $readme

# Detailed independent report in HTML; Edge converts this exact content to the required PDF.
$html = @'
<!doctype html><html><head><meta charset="utf-8"><title>NER Deliverable 2</title>
<style>
@page { size: A4; margin: 20mm; } body { font: 11pt Arial, sans-serif; color:#172033; line-height:1.45; }
h1 { color:#17365d; font-size:24pt; } h2 { color:#24527a; border-bottom:1px solid #9fbad0; padding-bottom:4px; }
h3 { color:#376b8f; } .cover { height:245mm; display:flex; flex-direction:column; justify-content:center; text-align:center; page-break-after:always; }
.cover p { font-size:14pt; } .toc { page-break-after:always; } table { border-collapse:collapse; width:100%; margin:10px 0 18px; font-size:9.5pt; }
th,td { border:1px solid #9aa8b4; padding:6px; text-align:left; } th { background:#dce8f2; }
.note { background:#eef5fa; border-left:4px solid #376b8f; padding:9px; } code { font-family:Consolas,monospace; }
</style></head><body>
<section class="cover"><h1>Named Entity Recognition</h1><h2>Deliverable 2</h2>
<p>Classic feature engineering and modern deep learning</p>
<p><strong>Team members</strong><br>Jose Calatayud<br>Miquel &Agrave;ngel Llauger<br>Lluc Segura</p>
<p>Natural Language Processing<br>Universitat de Barcelona<br>20 June 2026</p></section>
<section class="toc"><h2>2. Table of contents</h2><ol>
<li>Front page</li><li>Table of contents</li><li>Introduction</li><li>Exploratory data analysis</li>
<li>Methodology</li><li>Experimental setup</li><li>Results</li><li>Conclusions</li>
<li>Team member contributions</li><li>Use of AI</li></ol></section>

<h2>3. Introduction</h2>
<p>Named Entity Recognition (NER) assigns a semantic label to each token in a sentence, identifying people, locations, organisations, times and other entity types. It is a sequence-labelling problem used in information retrieval, search, question answering and knowledge extraction. This project compares the two approaches requested in the course guide: a classic conditional random field (CRF) based on handcrafted features and deep-learning systems that learn representations from data.</p>
<p>The work follows a controlled progression. Three CRFs show the effect of increasingly rich morphology and context. A BiLSTM removes manual feature engineering but exposes overfitting, out-of-vocabulary (OOV) handling and independent tag decisions. GloVe embeddings address lexical coverage, a CRF output layer models sequence transitions, their combination addresses both issues, and DistilBERT adds contextual subword representations.</p>

<h2>4. Exploratory data analysis</h2>
<p>The supplied token-level CSV files contain approximately 38,366 training sentences and 38,367 test sentences, with roughly 840,000 tokens in each large split. The TINY TEST contains 13 deliberately ambiguous or misspelled sentences. Tokens are grouped by <code>sentence_id</code>; every token has a word and a BIO tag.</p>
<p>There are 17 evaluation labels: <code>O</code> plus B/I variants for <code>geo</code>, <code>gpe</code>, <code>org</code>, <code>per</code>, <code>tim</code>, <code>art</code>, <code>eve</code> and <code>nat</code>. About 83% of tokens are <code>O</code>, so ordinary token accuracy would be misleading. Rare labels such as natural phenomena, events and artefacts occur only a few hundred times. Sentence-length inspection motivates a maximum sequence length of 60 for the recurrent models. The train/test vocabulary mismatch also motivates pretrained and subword representations.</p>

<h2>5. Methodology</h2>
<h3>5.1 Conditional random fields</h3>
<p>Three linear-chain CRFs were implemented with <code>sklearn-crfsuite</code>. Version 1 uses lowercase, case, title-case and digit indicators. Version 2 adds prefixes, suffixes, word shape and one-token context. Version 3 adds multi-length affixes, normalised shape, orthographic flags and context within two tokens, with adjusted L1/L2 regularisation. A CRF scores an entire label sequence, so learned transitions encourage valid BIO patterns.</p>
<h3>5.2 BiLSTM family</h3>
<p>The baseline PyTorch BiLSTM learns word embeddings from scratch and encodes left and right context before token-wise classification. The BiLSTM-CRF replaces independent softmax decisions with learned transition scores and Viterbi decoding, using validation-based early stopping and weight decay. The GloVe variant initialises a 300-dimensional embedding table from <code>glove-wiki-gigaword-300</code>, uses case-aware lowercase fallback and extends the vocabulary for improved OOV coverage. The combined GloVe-BiLSTM-CRF incorporates both pretrained lexical knowledge and structured decoding.</p>
<h3>5.3 DistilBERT</h3>
<p><code>distilbert-base-cased</code> was fine-tuned for token classification. Word labels are aligned to WordPiece tokens: the first subtoken receives the word label and later subtokens are ignored in the loss with <code>-100</code>. Contextual embeddings can distinguish uses such as Apple the organisation from apples the fruit, while subwords reduce the impact of unseen spellings.</p>

<h2>6. Experimental setup</h2>
<p>Random seeds are fixed to 42. Neural variants use a 90/10 training/validation split, Adam optimisation, dropout, weight decay and early stopping. Typical recurrent settings are hidden size 128, maximum length 60, batch size 64 and learning rate 0.001. DistilBERT uses a cased checkpoint, learning rate around 3e-5 and GPU-oriented full training; CPU fallback deliberately uses fewer samples and epochs.</p>
<p>Following the guide, the primary accuracy is computed only at positions whose gold tag is not <code>O</code>. Weighted token F1 excludes <code>O</code>, confusion matrices are produced for train and test, and the 13 TINY TEST predictions are printed as <code>word/tag</code> pairs together with non-O accuracy. The independent test split is never used to update parameters.</p>

<h2>7. Results</h2>
<table><tr><th>Model</th><th>Train non-O acc.</th><th>Test non-O acc.</th><th>Train weighted F1</th><th>Test weighted F1</th><th>Tiny acc.</th></tr>
<tr><td>CRF v1</td><td>0.864</td><td>0.477</td><td>0.876</td><td>0.477</td><td>0.765</td></tr>
<tr><td>CRF v2</td><td>0.931</td><td>0.705</td><td>0.934</td><td>0.727</td><td>0.882</td></tr>
<tr><td><strong>CRF v3</strong></td><td>0.963</td><td><strong>0.732</strong></td><td>0.965</td><td><strong>0.753</strong></td><td>0.765</td></tr>
<tr><td>BiLSTM</td><td>0.834</td><td>0.460</td><td>0.847</td><td>0.477</td><td>0.765</td></tr>
<tr><td>DistilBERT (CPU fallback)</td><td>0.759</td><td>0.585</td><td>0.762</td><td>0.590</td><td>0.882</td></tr></table>
<p>The strongest reported full test result is CRF v3 (weighted F1 0.753), and the increase from v1 to v3 demonstrates the value of feature engineering. CRF v2 and CPU-fallback DistilBERT obtain the best TINY TEST accuracy (0.882). The baseline BiLSTM generalises poorly compared with its training performance because learned embeddings collapse unseen words to <code>&lt;UNK&gt;</code> and tags are selected independently.</p>
<p>The additional BiLSTM-CRF, BiLSTM-GloVe and combined GloVe-BiLSTM-CRF experiments are retained in the training notebook and fitted-model directory. Their notebook runs show that pretrained embeddings improve lexical coverage and CRF decoding improves BIO consistency, but results depend strongly on whether the CPU smoke-test or full GPU configuration is used. For that reason, the table reports only the directly comparable values captured by the central reproduction run and does not mix smoke-test figures with full-data figures.</p>
<div class="note"><strong>Artifact note.</strong> The submitted DistilBERT directory contains configuration and tokenizer files but no model weight file. The preserved notebook outputs document its run, while fresh reproduction should skip it unless the weights are regenerated by <code>train_models.ipynb</code>. All supplied CRF and recurrent weight artifacts are included.</div>
<p>Qualitatively, the TINY TEST exposes ambiguity and robustness limits: London can be a surname or location, Apple can be an organisation or a common noun, and misspellings such as Barchelona, Parris and Microsof test generalisation beyond memorisation. Contextual and subword models are well suited to these cases, while CRF morphology remains surprisingly competitive.</p>

<h2>8. Conclusions</h2>
<p>The experiments confirm that a carefully engineered CRF is a strong NER baseline. Richer local context and morphology produced the best comparable test score without requiring a GPU. Deep learning removes manual feature design, but a from-scratch BiLSTM needs explicit safeguards against OOV collapse, overfitting and invalid output transitions. GloVe and a CRF head target these weaknesses separately and combine naturally. DistilBERT offers the most expressive contextual representation, but its result here is limited by fallback training and incomplete exported weights.</p>
<p>Future work should train all neural systems under one full-GPU protocol, retain every final checkpoint, tune CRF regularisation by validation search, report macro and entity-level F1 with <code>seqeval</code>, and analyse rare labels separately. These changes would improve both scientific comparability and reproducibility.</p>

<h2>9. Team member contributions</h2>
<p><strong>Jose Calatayud:</strong> implemented and documented the CRF progression, recurrent and Transformer experiments, fitted-model persistence, and consolidated reproduction workflow.</p>
<p><strong>Miquel &Agrave;ngel Llauger:</strong> contributed delivery structure, shared utility design, reproducibility checks and review of the evaluation requirements.</p>
<p><strong>Lluc Segura:</strong> contributed data/EDA review, report organisation, interpretation of results and final quality assurance.</p>

<h2>10. Use of AI</h2>
<p>Generative AI tools were used as development assistance for code review, debugging, documentation, restructuring notebooks to match the submission specification, and improving the clarity of this report. Team members inspected the generated material, retained responsibility for modelling choices and claims, and validated the final folder structure and notebook syntax. AI was not treated as an experimental result or an independent author.</p>
</body></html>
'@
Set-Content -Encoding UTF8 (Join-Path $dst 'main.html') $html

# Build an editable, standards-compliant DOCX containing the report text.
Add-Type -AssemblyName System.IO.Compression.FileSystem
$docx = Join-Path $dst 'main.docx'
if (Test-Path $docx) {
    try { Remove-Item $docx -Force }
    catch {
        $docx = Join-Path $dst 'main_corrected.docx'
        if (Test-Path $docx) { Remove-Item $docx -Force }
        Write-Warning 'main.docx is locked; wrote the refreshed report as main_corrected.docx.'
    }
}
$zip = [System.IO.Compression.ZipFile]::Open($docx, 'Create')
function Add-ZipText($zipFile, $name, $text) {
    $entry = $zipFile.CreateEntry($name)
    $writer = New-Object IO.StreamWriter($entry.Open(), [Text.UTF8Encoding]::new($false))
    $writer.Write($text); $writer.Dispose()
}
$types = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'
$rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'
$docrels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'
$styles = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:pPr><w:jc w:val="both"/><w:spacing w:after="140" w:line="276" w:lineRule="auto"/></w:pPr><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:sz w:val="22"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:pPr><w:jc w:val="center"/><w:spacing w:before="2600" w:after="240"/></w:pPr><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:b/><w:color w:val="17365D"/><w:sz w:val="52"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:pPr><w:jc w:val="center"/><w:spacing w:after="240"/></w:pPr><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:color w:val="376B8F"/><w:sz w:val="30"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Cover"><w:name w:val="Cover"/><w:pPr><w:jc w:val="center"/><w:spacing w:after="140"/></w:pPr><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:sz w:val="24"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:after="240"/></w:pPr><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:b/><w:color w:val="24527A"/><w:sz w:val="34"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="180" w:after="100"/></w:pPr><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:b/><w:color w:val="376B8F"/><w:sz w:val="26"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="TOC"><w:name w:val="TOC"/><w:pPr><w:ind w:left="360"/><w:spacing w:after="100"/></w:pPr><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:sz w:val="22"/></w:rPr></w:style></w:styles>'
Add-ZipText $zip '[Content_Types].xml' $types; Add-ZipText $zip '_rels/.rels' $rels
Add-ZipText $zip 'word/_rels/document.xml.rels' $docrels; Add-ZipText $zip 'word/styles.xml' $styles

$plain = $html -replace '(?is)<style.*?</style>', '' -replace '(?i)<br\s*/?>', "`n" -replace '(?i)</(h1|h2|h3|p|li|tr|div|section)>', "`n" -replace '(?i)<li>', '- ' -replace '(?i)<t[dh][^>]*>', ' | ' -replace '(?s)<[^>]+>', ''
$plain = [System.Net.WebUtility]::HtmlDecode($plain)
$paragraphs = New-Object Text.StringBuilder
$inCover = $true
$inTable = $false
$tableRows = New-Object Text.StringBuilder
foreach ($line in ($plain -split "`r?`n")) {
    $line = $line.Trim(); if (-not $line) { continue }
    if ($line.StartsWith('|')) {
        $inTable = $true
        $cells = $line.Trim('|').Split('|') | ForEach-Object { $_.Trim() }
        $isHeader = $cells[0] -eq 'Model'
        $widths = @(2200, 1360, 1360, 1440, 1440, 1200)
        $row = New-Object Text.StringBuilder
        for ($c = 0; $c -lt $cells.Count; $c++) {
            $cellText = [Security.SecurityElement]::Escape($cells[$c])
            $shade = if ($isHeader) { '<w:shd w:fill="DCE8F2"/>' } else { '' }
            $bold = if ($isHeader) { '<w:b/>' } else { '' }
            $align = if ($c -eq 0) { 'left' } else { 'center' }
            [void]$row.Append("<w:tc><w:tcPr><w:tcW w:w=`"$($widths[$c])`" w:type=`"dxa`"/>$shade</w:tcPr><w:p><w:pPr><w:jc w:val=`"$align`"/><w:spacing w:before=`"70`" w:after=`"70`"/></w:pPr><w:r><w:rPr>$bold<w:sz w:val=`"18`"/></w:rPr><w:t>$cellText</w:t></w:r></w:p></w:tc>")
        }
        [void]$tableRows.Append("<w:tr>$($row.ToString())</w:tr>")
        continue
    }
    if ($inTable) {
        $tablePr = '<w:tblPr><w:tblW w:w="9000" w:type="dxa"/><w:tblLayout w:type="fixed"/><w:tblBorders><w:top w:val="single" w:sz="8" w:color="7F8C99"/><w:left w:val="single" w:sz="8" w:color="7F8C99"/><w:bottom w:val="single" w:sz="8" w:color="7F8C99"/><w:right w:val="single" w:sz="8" w:color="7F8C99"/><w:insideH w:val="single" w:sz="6" w:color="9AA8B4"/><w:insideV w:val="single" w:sz="6" w:color="9AA8B4"/></w:tblBorders></w:tblPr>'
        [void]$paragraphs.Append("<w:tbl>$tablePr$($tableRows.ToString())</w:tbl>")
        $inTable = $false; $tableRows.Clear() | Out-Null
    }
    $style = 'Normal'
    $extra = ''
    if ($line -eq 'Named Entity Recognition') { $style = 'Title' }
    elseif ($line -eq 'Deliverable 2') { $style = 'Subtitle' }
    elseif ($line -match '^2\. Table of contents$') { $style = 'Heading1'; $extra = '<w:pageBreakBefore/>'; $inCover = $false }
    elseif ($line -match '^(3|4|5|6|7|8|9|10)\. ') { $style = 'Heading1'; $extra = '<w:pageBreakBefore/>' }
    elseif ($line -match '^\d+\.\d+ ') { $style = 'Heading2' }
    elseif ($inCover) { $style = 'Cover' }
    elseif ($line.StartsWith('- ')) { $style = 'TOC' }
    $escaped = [Security.SecurityElement]::Escape($line)
    [void]$paragraphs.Append("<w:p><w:pPr><w:pStyle w:val=`"$style`"/>$extra</w:pPr><w:r><w:t xml:space=`"preserve`">$escaped</w:t></w:r></w:p>")
}
if ($inTable) {
    $tablePr = '<w:tblPr><w:tblW w:w="9000" w:type="dxa"/><w:tblLayout w:type="fixed"/><w:tblBorders><w:top w:val="single" w:sz="8" w:color="7F8C99"/><w:left w:val="single" w:sz="8" w:color="7F8C99"/><w:bottom w:val="single" w:sz="8" w:color="7F8C99"/><w:right w:val="single" w:sz="8" w:color="7F8C99"/><w:insideH w:val="single" w:sz="6" w:color="9AA8B4"/><w:insideV w:val="single" w:sz="6" w:color="9AA8B4"/></w:tblBorders></w:tblPr>'
    [void]$paragraphs.Append("<w:tbl>$tablePr$($tableRows.ToString())</w:tbl>")
}
$document = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>' + $paragraphs.ToString() + '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/></w:sectPr></w:body></w:document>'
Add-ZipText $zip 'word/document.xml' $document
$zip.Dispose()

# Generate a styled PDF directly so the build does not depend on Word or a browser.
function Wrap-ReportLine([string]$line, [int]$width = 92) {
    if ($line.Length -le $width) { return @($line) }
    $result = New-Object System.Collections.ArrayList
    $current = ''
    foreach ($word in ($line -split '\s+')) {
        if (-not $current) { $current = $word; continue }
        if (($current.Length + 1 + $word.Length) -le $width) { $current += " $word" }
        else { [void]$result.Add($current); $current = $word }
    }
    if ($current) { [void]$result.Add($current) }
    return $result.ToArray()
}

# Split the report into the ten sections required by the guide. Each starts on a new page.
$pageGroups = New-Object System.Collections.ArrayList
$group = New-Object System.Collections.ArrayList
foreach ($line in ($plain -split "`r?`n")) {
    $line = ($line -replace '\s+', ' ').Trim()
    if (-not $line) { continue }
    if ($line -match '^(2\. Table of contents|3\. Introduction|4\. Exploratory|5\. Methodology|6\. Experimental|7\. Results|8\. Conclusions|9\. Team|10\. Use)') {
        if ($group.Count -gt 0) { [void]$pageGroups.Add($group); $group = New-Object System.Collections.ArrayList }
    }
    [void]$group.Add($line)
}
if ($group.Count -gt 0) { [void]$pageGroups.Add($group) }

function Escape-PdfText([string]$text) {
    return $text.Replace('\', '\\').Replace('(', '\(').Replace(')', '\)')
}

$pdfCommands = New-Object System.Collections.ArrayList
for ($p = 0; $p -lt $pageGroups.Count; $p++) {
    $commands = New-Object Text.StringBuilder
    $y = if ($p -eq 0) { 620 } else { 785 }
    $previousWasTable = $false
    foreach ($line in $pageGroups[$p]) {
        if ($line.StartsWith('|')) {
            $cells = $line.Trim('|').Split('|') | ForEach-Object { $_.Trim() }
            $columnX = @(38, 175, 250, 325, 405, 485, 557)
            $rowHeight = 24
            $isHeader = $cells[0] -eq 'Model'
            for ($c = 0; $c -lt $cells.Count; $c++) {
                $cellX = $columnX[$c]; $cellWidth = $columnX[$c + 1] - $cellX
                if ($isHeader) { [void]$commands.Append("0.86 0.91 0.95 rg $cellX $($y - $rowHeight) $cellWidth $rowHeight re f`n") }
                [void]$commands.Append("0.45 0.52 0.58 RG $cellX $($y - $rowHeight) $cellWidth $rowHeight re S`n")
                $cellFont = if ($isHeader) { 'F2' } else { 'F1' }
                $cellSize = if ($c -eq 0) { 6.5 } else { 6.2 }
                $cellText = Escape-PdfText $cells[$c]
                [void]$commands.Append("0 g BT /$cellFont $cellSize Tf 1 0 0 1 $($cellX + 4) $($y - 15) Tm ($cellText) Tj ET`n")
            }
            $y -= $rowHeight
            $previousWasTable = $true
            continue
        }
        if ($previousWasTable) {
            $y -= 20
            $previousWasTable = $false
        }
        $font = 'F1'; $size = 10; $x = 50; $leading = 14; $wrap = 92
        if ($line -eq 'Named Entity Recognition') { $font = 'F2'; $size = 26; $x = 145; $leading = 38; $wrap = 60 }
        elseif ($line -eq 'Deliverable 2') { $font = 'F2'; $size = 18; $x = 220; $leading = 30; $wrap = 65 }
        elseif ($p -eq 0) { $size = 12; $x = 145; $leading = 22; $wrap = 58 }
        elseif ($line -match '^\d+\.\d+ ') { $font = 'F2'; $size = 13; $x = 50; $leading = 23; $wrap = 78 }
        elseif ($line -match '^\d+\. ') { $font = 'F2'; $size = 19; $x = 50; $leading = 34; $wrap = 65 }
        elseif ($line.StartsWith('- ')) { $x = 75; $size = 11; $leading = 18; $wrap = 76 }
        foreach ($wrapped in (Wrap-ReportLine $line $wrap)) {
            $escaped = Escape-PdfText $wrapped
            [void]$commands.Append("BT /$font $size Tf 1 0 0 1 $x $y Tm ($escaped) Tj ET`n")
            $y -= $leading
        }
        if ($line -match '^\d+\. ') {
            [void]$commands.Append("0.25 0.45 0.65 RG 50 $($y + 15) m 545 $($y + 15) l S`n")
        }
        if ($line -match '^\d+\.\d+ ') { $y -= 3 }
        elseif (-not ($line -match '^\d+\. ')) { $y -= 5 }
    }
    [void]$commands.Append("BT /F1 8 Tf 0.35 g 1 0 0 1 278 25 Tm ($($p + 1) / $($pageGroups.Count)) Tj ET`n")
    [void]$pdfCommands.Add($commands.ToString())
}

$pageCount = $pdfCommands.Count
$encoding = [Text.Encoding]::GetEncoding(1252)
$objects = @{}
$objects[1] = '<< /Type /Catalog /Pages 2 0 R >>'
$pageIds = for ($p = 0; $p -lt $pageCount; $p++) { 6 + (2 * $p) }
$objects[2] = '<< /Type /Pages /Kids [' + (($pageIds | ForEach-Object { "$_ 0 R" }) -join ' ') + "] /Count $pageCount >>"
$objects[3] = '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>'
$objects[4] = '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>'
for ($p = 0; $p -lt $pageCount; $p++) {
    $contentId = 5 + (2 * $p); $pageId = 6 + (2 * $p)
    $stream = [string]$pdfCommands[$p]
    $streamLength = $encoding.GetByteCount($stream)
    $objects[$contentId] = "<< /Length $streamLength >>`nstream`n$stream`nendstream"
    $objects[$pageId] = "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents $contentId 0 R >>"
}
$maxObject = 4 + (2 * $pageCount)
$builder = New-Object Text.StringBuilder
[void]$builder.Append("%PDF-1.4`n")
$offsets = New-Object System.Collections.ArrayList
[void]$offsets.Add(0)
for ($id = 1; $id -le $maxObject; $id++) {
    [void]$offsets.Add($encoding.GetByteCount($builder.ToString()))
    [void]$builder.Append("$id 0 obj`n$($objects[$id])`nendobj`n")
}
$xrefOffset = $encoding.GetByteCount($builder.ToString())
[void]$builder.Append("xref`n0 $($maxObject + 1)`n0000000000 65535 f `n")
for ($id = 1; $id -le $maxObject; $id++) {
    [void]$builder.Append(('{0:D10} 00000 n ' -f [int]$offsets[$id]) + "`n")
}
[void]$builder.Append("trailer`n<< /Size $($maxObject + 1) /Root 1 0 R >>`nstartxref`n$xrefOffset`n%%EOF`n")
[IO.File]::WriteAllBytes((Join-Path $dst 'main.pdf'), $encoding.GetBytes($builder.ToString()))

Write-Output "Built delivery at $dst"
