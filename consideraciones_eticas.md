# Consideraciones Éticas y Responsabilidad Social del Proyecto

## 1. Análisis de sesgos
El dataset utilizado fue recopilado a partir de fuentes abiertas y hospitales locales, lo cual implica la posibilidad de sesgos demográficos y técnicos.  
Pueden existir **diferencias en la representación por edad, género, etnia o calidad de los equipos ecográficos**, lo que podría afectar la generalización del modelo.  
Estos sesgos pueden **favorecer o perjudicar la precisión en ciertos grupos poblacionales**, por ejemplo, en pacientes jóvenes o con menor densidad tiroidea.  
Para mitigar estos riesgos, se realizan auditorías de diversidad de datos y evaluaciones periódicas de *fairness metrics*.

## 2. Equidad y fairness
El modelo busca tratar a todos los grupos de forma equitativa mediante técnicas de balanceo y normalización del dataset.  
Se aplican **métricas de equidad** como *Demographic Parity Difference* y *Equal Opportunity Ratio*.  
En caso de detectar inequidades significativas, se ajustan ponderaciones o se realizan estrategias de *re-sampling* para garantizar la equidad intergrupal.  
Estas medidas permiten reducir disparidades sin comprometer la sensibilidad médica del modelo.

## 3. Privacidad
El proyecto no utiliza información personal identificable.  
Las imágenes ecográficas fueron **anonimizadas** eliminando nombres, fechas y metadatos de procedencia.  
La privacidad se protege mediante **encriptación AES-256** durante el almacenamiento y transmisión de datos.  
Además, se cumple con las normativas **GDPR (Europa)**, **HIPAA (EE.UU.)** y **Ley Orgánica de Protección de Datos Personales (Ecuador)**.  
El acceso a los datos está restringido mediante autenticación y control de roles.

## 4. Transparencia y explicabilidad
El modelo es interpretado a través de técnicas de **Explainable AI**, principalmente **Grad-CAM**, que permite visualizar las regiones de la ecografía que influyen en la decisión.  
Se han implementado **reportes interpretativos automáticos** para que los profesionales médicos comprendan cómo el modelo llega a una predicción.  
El uso de *notebooks reproducibles* y documentación técnica garantiza la trazabilidad de resultados y modelos entrenados.

## 5. Impacto social
El proyecto promueve la **democratización del acceso a diagnósticos especializados**, especialmente en regiones con limitaciones de personal médico.  
Entre los impactos positivos destacan la **detección temprana del cáncer de tiroides**, reducción de errores diagnósticos y optimización de recursos hospitalarios.  
Sin embargo, podría generar dependencia tecnológica o **desigualdad digital** si solo algunos hospitales acceden a la tecnología.  
Los beneficiarios principales son pacientes y profesionales de salud; los potencialmente perjudicados serían grupos con acceso limitado a infraestructura digital.

## 6. Responsabilidad
En caso de error, la responsabilidad recae en un **marco compartido** entre el equipo técnico y los especialistas médicos que supervisan el diagnóstico final.  
Se ha establecido un **plan de monitoreo y trazabilidad** de versiones de modelos, con un comité ético-médico encargado de auditar incidentes.  
Cada versión del sistema incluye registro de fecha, responsables, métricas de desempeño y validaciones realizadas.

## 7. Uso dual y mal uso
Existe el riesgo potencial de **uso indebido del modelo** fuera del ámbito clínico.  
Para prevenirlo, se restringe el acceso al código entrenado y se documentan las limitaciones de uso.  
El modelo no debe aplicarse en otros tipos de cáncer ni en ecografías fuera de la región tiroidea sin validación científica previa.  
Asimismo, se establecen políticas de licencia y supervisión que impiden su empleo con fines comerciales no autorizados.

## 8. Limitaciones reconocidas
El modelo no debe utilizarse como herramienta de diagnóstico autónomo ni sustituir el criterio médico.  
Su rendimiento puede verse afectado por imágenes de baja calidad, equipos ecográficos no calibrados o poblaciones no representadas en el dataset.  
Se advierte a los usuarios sobre las **condiciones bajo las cuales el modelo no es confiable**, garantizando una aplicación responsable y transparente.

## 9. Conclusión
El proyecto mantiene un compromiso ético integral con los principios de **justicia, equidad, transparencia, responsabilidad y privacidad**.  
La incorporación de procesos de auditoría, documentación interpretativa y revisión médica constante asegura que la inteligencia artificial se aplique de manera responsable y en beneficio de la salud pública.  
El equilibrio entre la innovación tecnológica y la ética médica constituye la base fundamental de este desarrollo.
