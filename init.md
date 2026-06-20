# Init Para Agentes

## Nota Para Agentes En Este VPS

Este VPS tiene recursos limitados. No ejecutes pruebas, entrenamientos, builds pesados ni comandos de verificacion largos en este entorno, aunque existan instrucciones de testing en la documentacion del proyecto. Haz cambios de codigo con revision estatica ligera y deja que las pruebas se ejecuten en una maquina local con mejor hardware.

Para Mundial, el flujo vigente usa solo el dataset internacional `all_matches.csv` desde 2014. El ETL debe dividir por tiempo en 80/10/10: train inicial, validacion intermedia y test final. El entrenamiento de boosting queda en perfil de features `balanced` por defecto: maximo 480 columnas, sin `train.csv/test.csv`, sin familia `kaggle_`, con `history` compacto y ventanas 3/5/10. Si un agente toca esta parte, debe conservar esos defaults salvo que el usuario pida explicitamente modo completo.

Cuando termines cambios en este repositorio, commitea y sube todo a Git:

```bash
git add <archivos modificados>
git commit -m "mensaje claro"
git push origin main
```
