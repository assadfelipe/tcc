select mes, tipo_crime, latitude, longitude, upper(peridoocorrencia) from furto_veiculo_teste
where mes <> '' and mes <> 'continuous' and latitude <> ''


select mes, tipo_crime, latitude, longitude, upper(peridoocorrencia) from roubo_veiculo_teste
where mes <> '' and mes <> 'continuous' and latitude <> ''

select mes, tipo_crime, latitude, longitude, upper(peridoocorrencia) from latrocinio_teste
where mes <> '' and mes <> 'continuous' and latitude <> ''

select mes, tipo_crime, latitude, longitude, upper(peridoocorrencia) from lesao_corporal_teste
where mes <> '' and mes <> 'continuous' and latitude <> ''

select mes, tipo_crime, latitude, longitude, upper(peridoocorrencia) from intervencao_policial_teste
where mes <> '' and mes <> 'continuous' and latitude <> ''


select count(*) from roubo_veiculo_treinamento where mes <> '' and mes <> 'discrete' and latitude <> ''