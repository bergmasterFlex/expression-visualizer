#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct FunctionDeclarationId(pub usize);

#[derive(Clone)]
pub struct FunctionDeclaration {
    pub name: String,
    pub inputs: Vec<FunctionParameterDeclaration>,
    pub output_type: crate::eval::EType,
}

#[derive(Clone)]
pub struct FunctionParameterDeclaration {
    pub name: String,
    pub r#type: crate::eval::EType,
}
