#[derive(Debug, Clone)]
pub enum EType {
    Int(Option<i32>),
    Float(Option<f32>),
    Bool(Option<bool>),
    String(Option<String>),
    Char(Option<char>),
    Any,
    Undefined,
    Exception,
    SumType(Vec<EType>),
}

impl ToString for EType {
    fn to_string(&self) -> String {
        match self {
            EType::Int(value) => {
                if let Some(value) = value {
                    value.to_string()
                } else {
                    "int".to_string()
                }
            }
            EType::Float(value) => {
                if let Some(value) = value {
                    value.to_string()
                } else {
                    "float".to_string()
                }
            }
            EType::Bool(value) => {
                if let Some(value) = value {
                    value.to_string()
                } else {
                    "bool".to_string()
                }
            }
            EType::String(value) => {
                if let Some(value) = value {
                    value.to_string()
                } else {
                    "string".to_string()
                }
            }
            EType::Char(value) => {
                if let Some(value) = value {
                    value.to_string()
                } else {
                    "char".to_string()
                }
            }
            EType::Any => "any".to_string(),
            EType::Undefined => "undefined".to_string(),
            EType::Exception => "exception".to_string(),
            EType::SumType(sub_types) => sub_types
                .iter()
                .map(|sub_type| sub_type.to_string())
                .collect::<Vec<_>>()
                .join("|"),
        }
    }
}
