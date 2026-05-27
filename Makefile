.PHONY: informe informe-clean

# Genera el PDF del informe.
# Uso: make informe
informe:
	./informe/generar_pdf.sh

# Limpia archivos auxiliares de LaTeX y regenera el PDF desde cero.
# Uso: make informe-clean
informe-clean:
	./informe/generar_pdf.sh --clean
