# LaTeX Local Setup and PDF Generation Guide (macOS)

This guide explains how to install LaTeX locally on macOS and generate PDFs from a `.tex` project without using Overleaf.

---

## 1. Install MacTeX

Download and install MacTeX:

https://www.tug.org/mactex/

This installs:

- `pdflatex`
- `latexmk`
- BibTeX
- common LaTeX packages

---

## 2. Add LaTeX binaries to PATH

After installation, open a terminal and run:

```bash
echo 'export PATH="/Library/TeX/texbin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Verify installation:

```bash
latexmk -v
```

Expected output example:

```bash
Latexmk, John Collins, Version 4.xx
```

---

## 3. Open the LaTeX project

Navigate to the project folder containing `main.tex`.

In this project, the folder name is `informe`.

```bash
cd informe
```

Verify the file exists:

```bash
ls
```

You should see something like:

```bash
main.tex
```

---

## 4. Generate the PDF

Run:

```bash
latexmk -pdf main.tex
```

This generates:

```bash
main.pdf
```

---

## 5. Open the generated PDF

On macOS:

```bash
open main.pdf
```

---

## 6. Force a clean rebuild

If LaTeX gets stuck with cached errors or outdated files:

```bash
latexmk -C
rm -f main.pdf
latexmk -pdf main.tex
```

This:

- removes auxiliary/generated files
- removes the old PDF
- recompiles everything from scratch

---

## 7. Re-generate the PDF after making changes

After editing the `.tex` files:

```bash
latexmk -pdf main.tex
```

Then reopen:

```bash
open main.pdf
```

---

## 8. Auto-recompile on every save

This behaves similarly to Overleaf live compilation:

```bash
latexmk -pvc -pdf main.tex
```

Options:

- `-pvc` = preview continuously
- automatically recompiles when files change

Keep this terminal running while editing.

---

## 9. Common errors

### Missing image

Example:

```bash
File `figures/example.png' not found
```

Check:

- the file exists
- the filename matches exactly
- uppercase/lowercase matches
- the path is correct

---

### Missing bibliography file

Example:

```bash
Bib file(s) not found:
Libros.bib
```

Check that:

- the `.bib` file exists
- the filename matches exactly
- the bibliography command points to the correct file

Example:

```latex
\bibliography{Libros}
```

expects a file named:

```bash
Libros.bib
```

---

### Missing table of contents (`main.toc`)

Example:

```bash
No file main.toc.
```

This is usually **not a real error**.

The table of contents is generated after a successful compilation. If LaTeX fails before finishing, the `.toc` file is never created.

Once the project compiles correctly, rerun:

```bash
latexmk -pdf main.tex
```

and the index/table of contents should appear.

---

### PDF generated but empty

Usually caused by a LaTeX compilation error.

See the latest errors with:

```bash
tail -80 main.log
```

---

### Missing figures from the project

Example:

```bash
File `figures/Foraging/jalam2iql_config2_training.png' not found
```

Solution:

- verify the image exists
- verify uppercase/lowercase
- verify the path is correct
- temporarily comment the `\includegraphics{}` line if needed

Example:

```latex
% \includegraphics[width=\linewidth]{figures/Foraging/jalam2iql_config2_training.png}
```

---

## 10. Useful commands

### Clean generated files

```bash
latexmk -C
```

### Compile PDF

```bash
latexmk -pdf main.tex
```

### Open PDF

```bash
open main.pdf
```

### Watch and auto-compile

```bash
latexmk -pvc -pdf main.tex
```

### View recent errors

```bash
tail -80 main.log
```

### Check generated PDF size

```bash
ls -lh main.pdf
```

If the size is `0B`, the PDF is empty because LaTeX failed before generating it correctly.

---

## 11. Recommended editor

Recommended editors:

- VS Code
- Cursor

Recommended extension:

- LaTeX Workshop

This provides:

- syntax highlighting
- live PDF preview
- auto-build support
- error navigation
