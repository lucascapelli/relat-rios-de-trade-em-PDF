import boto3
import os
from botocore.exceptions import NoCredentialsError, ClientError
from app.utils import logger
from app.config import (
    AWS_ACCESS_KEY_ID, 
    AWS_SECRET_ACCESS_KEY, 
    AWS_BUCKET_NAME, 
    AWS_REGION,
    AWS_ENDPOINT_URL
)

def get_s3_client():
    try:
        if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_BUCKET_NAME]):
            # Em desenvolvimento local, se não tiver credenciais, retorna None silenciosamente
            # ou avisa apenas uma vez. Aqui vamos logar e retornar None.
            return None
            
        client_kwargs = {
            'service_name': 's3',
            'aws_access_key_id': AWS_ACCESS_KEY_ID,
            'aws_secret_access_key': AWS_SECRET_ACCESS_KEY,
            'region_name': AWS_REGION
        }
        
        # Se for um serviço compatível (Backblaze, Cloudflare R2), usa o endpoint customizado
        if AWS_ENDPOINT_URL:
            client_kwargs['endpoint_url'] = AWS_ENDPOINT_URL

        return boto3.client(**client_kwargs)
    except Exception as e:
        logger.error(f"Erro ao criar cliente S3: {e}")
        return None

def get_presigned_url(object_name: str, expiration=3600) -> str:
    """
    Gera uma URL assinada temporária para um objeto privado.
    Expiration padrão de 1 hora para o 'Proxy' (link dinâmico).
    """
    s3_client = get_s3_client()
    if not s3_client:
        return None
        
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': AWS_BUCKET_NAME, 'Key': object_name},
            ExpiresIn=expiration
        )
        return url
    except ClientError as e:
        logger.error(f"Erro ao gerar URL assinada: {e}")
        return None

def upload_to_s3(file_path: str, object_name: str = None) -> str:
    """
    Upload a file to an S3 bucket and return the application proxy URL.
    Deletes the local file after successful upload.
    """
    s3_client = get_s3_client()
    if not s3_client:
        return None

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_path)

    try:
        # Upload como privado
        s3_client.upload_file(
            file_path, 
            AWS_BUCKET_NAME, 
            object_name, 
            ExtraArgs={'ContentType': 'application/pdf'}
        )
        
        # Remove arquivo local para economizar espaço (ephemeral filesystem)
        try:
            os.remove(file_path)
            logger.info(f"Arquivo local removido: {file_path}")
        except Exception as e:
            logger.warning(f"Não foi possível remover arquivo local: {e}")

        # Retorna a URL do PROXY (rota do Flask)
        # Isso garante que o link seja "eterno" (sempre gera uma nova assinatura ao acessar)
        proxy_url = f"/reports/{object_name}"
        
        logger.info(f"Arquivo enviado para S3. Proxy URL: {proxy_url}")
        
        return proxy_url
        
    except FileNotFoundError:
        logger.error(f"O arquivo {file_path} não foi encontrado.")
        return None
    except NoCredentialsError:
        logger.error("Credenciais AWS não disponíveis.")
        return None
    except ClientError as e:
        logger.error(f"Erro no cliente S3: {e}")
        return None
    except Exception as e:
        logger.error(f"Erro genérico ao enviar para S3: {e}")
        return None
